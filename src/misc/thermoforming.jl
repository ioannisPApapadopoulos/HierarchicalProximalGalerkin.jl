struct Thermoforming2D{V}
    r::AbstractVector{V}
    pT::Integer
    pu::Integer
    k::V
    Φ₀x::AbstractArray{V}
    ϕx::AbstractArray{V}
    g::Function
    dg::Function
    A::AbstractMatrix{V}
    M::AbstractMatrix{V}
    plan_C::PiecewiseOrthogonalPolynomials.ContinuousPolynomialTransform{V}
    plan_D::PiecewiseOrthogonalPolynomials.DirichletPolynomialTransform{V}
    xy::AbstractMatrix{V}
    constant_dg::Bool
    lu_G
end

function Thermoforming2D(r::AbstractVector{V}, pT::Integer, pu::Integer, k::V, Φ₀::Function, ϕ::Function, g::Function, dg::Function; constant_dg::Bool=false, lu_G=[]) where V
    @assert pT == pu
    C = ContinuousPolynomial{1}(r)
    Dp = DirichletPolynomial(r)
    xy, plan_C = plan_grid_transform(C, Block(pT+1, pT+1))
    plan_D = plan_transform(Dp, Block(pu+1, pu+1))
    x,y = first(xy), last(xy)
    Φ₀x = Φ₀.(x,reshape(y,1,1,size(y)...))
    ϕx = ϕ.(x,reshape(y,1,1,size(y)...))

    M1 = sparse(Symmetric(grammatrix(C)[Block.(1:pT+1), Block.(1:pT+1)]))
    M = Symmetric(kron(M1, M1))

    KR = Block.(oneto(pT+1))
    Δ = weaklaplacian(C)
    A1 = sparse(Symmetric(-parent(Δ)[KR,KR]))
    A = Symmetric(kron(A1,M1) + kron(M1,A1))

    if constant_dg
        dgx = dg.(Φ₀x)
        X = _MϕX(plan_C, M, dgx, ϕx, pT, length(r)-1)
        G =  sparse(A + k*M - X)
        lu_G = MatrixFactorizations.lu(G)
    else
        G =  sparse(A + k*M)
        lu_G = MatrixFactorizations.cholesky(G)
    end
    Thermoforming2D{V}(r, pT, pu, k, Φ₀x, ϕx, g, dg, A, M, plan_C, plan_D, xy[1], constant_dg, lu_G)
end

function reshapeD(TF::Thermoforming2D{V}, c::AbstractVector{V}) where V
    nx = TF.pu
    px = length(TF.r)-1
    BlockMatrix(reshape(c, (nx+1)*px-1, (nx+1)*px-1), [px-1; repeat([px], nx)], [px-1; repeat([px], nx)])
end

function reshapeC(TF::Thermoforming2D{V}, c::AbstractVector{V}) where V
    nx = TF.pT
    px = length(TF.r)-1
    BlockMatrix(reshape(c, (nx+1)*px+1, (nx+1)*px+1), [px+1; repeat([px], nx)], [px+1; repeat([px], nx)])
end

function evaluate_g(TF::Thermoforming2D{V}, T::AbstractVector{V}, u::AbstractVector{V}) where V
    plan_C, plan_D = TF.plan_C, TF.plan_D
    ux = plan_D \ reshapeD(TF, u)
    Tx = plan_C \ reshapeC(TF, T)
    vals = TF.g.(TF.Φ₀x .+ TF.ϕx .* Tx .- ux)
    TF.M * (plan_C * vals)[:]
end

function _MϕX(plan_C, M::AbstractMatrix{V}, dgx::AbstractArray{V}, ϕx::AbstractArray{V}, nx::Integer, px::Integer) where V
    X = ExtendableSparseMatrix(size(M)...)
    #TODO: the number of plans can be heavily reduced but since we only assemble this matrix once
    # the fix is not a priority.
    for j in 1:lastindex(M,2)
        X[:,j] = _apply_dg(plan_C,  M, dgx, Float64.(Vector(1:lastindex(M,2) .==j)), ϕx, nx, px)
    end
    return sparse(X)
end
function _apply_dg(plan_C, M::AbstractMatrix{V}, dgx::AbstractArray{V}, c::AbstractVector{V}, ϕx::AbstractArray{V}, nx::Integer, px::Integer) where V
    C = BlockMatrix(reshape(c, (nx+1)*px+1, (nx+1)*px+1), [px+1; repeat([px], nx)], [px+1; repeat([px], nx)])
    cx = plan_C \ C
    vals = dgx .* ϕx .* cx
    M * (plan_C * vals)[:]
end

function apply_dg(TF::Thermoforming2D{V}, dgx::AbstractArray{V}, c::AbstractVector{V}) where V
    plan_C = TF.plan_C
    cx = plan_C \ reshapeC(TF, c)
    vals = dgx .* TF.ϕx .* cx
    TF.M * (plan_C * vals)[:]
end

function residual(TF::Thermoforming2D{V}, T::AbstractVector{V}, u::AbstractVector{V}) where V
    A, M, k = TF.A, TF.M, TF.k
    A*T + k*M*T - evaluate_g(TF, T, u)
end 

function assembly_linear_solve(TF::Thermoforming2D{V}, b::AbstractVector{V}) where V
    @assert TF.constant_dg==true
    x = ldiv!(TF.lu_G, b)
    x
end

function matrixfree_linear_solve(TF::Thermoforming2D{V}, T::AbstractVector{V}, ux::AbstractArray{V}, b::AbstractVector{V};bandw::Integer=2, maxiter::Integer=200, tol::Float64=1e-10, verbosity::Integer=0, show_trace::Bool=true) where V

    Tx = TF.plan_C \ reshapeC(TF, T)
    dgx = TF.dg.(TF.Φ₀x .+ TF.ϕx .* Tx .- ux)

    A, M, k = TF.A, TF.M, TF.k
    n = size(TF.M,1)
    Sf(x) = Vector(A*x + k*M*x  - apply_dg(TF, dgx, x))
    S = LinearMap(Sf, n; ismutating=false)

    x, info = IterativeSolvers.gmres(S, b, Pl=TF.lu_G, log=true, restart=n)
    show_trace && print("GMRES Its: $(info.iters).\n")
    (x, info)
end

function solve(TF::Thermoforming2D{V}, u::AbstractVector{V}; T0=[], its_max::Integer=100, TOL::V=1e-10, show_trace::Bool=true, matrixfree::Bool=true) where V
    n = size(TF.M,1)
    T = isempty(T0) ? zeros(n) : copy(T0)
    gmres_iters = 0
    newton_iters = 0

    ux = TF.plan_D \ reshapeD(TF, u)

    ls_α =1

    for iter in 1:its_max
        res = residual(TF, T, u)
        normres = norm(res)
        show_trace && print("Iteration $(iter-1), stepsize: $ls_α, residual norm: $normres.\n")
        if normres < TOL
            break
        end
        if ls_α < 10000*eps()
            print("Linesearch stepsize below eps(), terminating loop. \n")
            break
        end

        if matrixfree
            (dx, info)= matrixfree_linear_solve(TF,T,ux,-res,show_trace=show_trace)
        else
            dx = assembly_linear_solve(TF,-res)
        end

        if matrixfree
            gmres_iters+=info.iters
        end
        newton_iters += 1
        (T,ls_α) = (T+dx, 1.0)
        # (T,ls_α) = backtracking == true ? pg_linesearch(PG,ls,u,ψ,w,α,du,dψ) : (T+dx, 1.0)
    end
    return T, (newton_iters, gmres_iters)
end