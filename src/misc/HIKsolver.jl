# 1D & 2D HIK solver
struct HIKSolver{T}
    A::AbstractMatrix{T}
    r::AbstractVector{T}
    Nh::Integer
    M::AbstractMatrix{T}
    Dp::DirichletPolynomial
    plan_D::PiecewiseOrthogonalPolynomials.DirichletPolynomialTransform{T}
    f::AbstractVector{T}
    φ::AbstractVector{T}
    d::Integer
end

function HIKSolver(r::AbstractVector{T}, f::Function, φ::Function) where T
    Nh = length(r)-1
    Dp = DirichletPolynomial(r)
    x, plan_D = plan_grid_transform(Dp, (Block(1),), 1)
    fv = Vector(plan_D * f.(x))
    φv = Vector(plan_D * φ.(x))

    Δ = weaklaplacian(Dp)
    A = sparse(Symmetric(-parent(Δ)[Block.(oneto(1)),Block.(oneto(1))]))
    M = sparse(grammatrix(Dp)[Block.(oneto(1)), Block.(oneto(1))])

    HIKSolver{T}(A, r, Nh, M, Dp, plan_D, fv, φv, 1)
end

function HIKSolver2D(r::AbstractVector{T}, f::Function, φ::Function) where T
    Nh = length(r)-1
    Dp = DirichletPolynomial(r)
    xy, plan_D = plan_grid_transform(Dp, Block(1,1))
    x,y = first(xy), last(xy)
    fv = (plan_D * f.(x,reshape(y,1,1,size(y)...)))[:]
    φv = (plan_D * φ.(x,reshape(y,1,1,size(y)...)))[:]

    Δ = weaklaplacian(Dp)
    A1 = sparse(Symmetric(-parent(Δ)[Block(1),Block(1)]))
    M1 = sparse(Symmetric((Dp' * Dp)[Block(1),Block(1)]))
    A = sparse(Symmetric(kron(A1,M1) + kron(M1,A1)))
    M = sparse(Symmetric(kron(M1, M1)))

    HIKSolver{T}(A, r, Nh, M, Dp, plan_D, fv, φv, 2)
end

function project!(x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where T
    b = x;
    b[x .< lb] = lb[x .< lb]
    b[x .> ub] = ub[x .> ub]
    return b
end

function reduced_residual(r::AbstractVector{T}, x::AbstractVector{T}, lb::AbstractVector{T}, ub::AbstractVector{T}) where T
    rr = r[:];
    rr[x .<= lb] = min.(rr[x .<= lb], zero(T))
    rr[x .>= ub] = max.(rr[x .>= ub], zero(T))
    return rr
end

function solve(HIK::HIKSolver{T}, u0::AbstractVector{T}=T[]; tol::T=1e-10,max_iter::Integer=10_000_000,damping::T=1.0,show_trace::Bool=true) where T
     
    u = isempty(u0) ? zeros( (HIK.Nh-1)^HIK.d )  : u0
    J, M, f, n = HIK.A, HIK.M, HIK.f, (HIK.Nh-1)^HIK.d
    Mf = M*f
    lb = -1e100*ones(lastindex(u))
    ub = HIK.φ

    index = Vector(1:lastindex(u))
    iter = 0
    inactive = index

    active_lb = []
    active_ub = []
    active    = []
    
    project!(u, lb, ub)

    r = J*u - Mf

    active_lb = findall(u .≈ lb) ∩ findall(r .> 0)
    active_ub = findall(u .≈ ub) ∩ findall(r .< 0)
    active = vcat(active_lb, active_ub)

    dual = zeros(T, n)
    dual[active_lb] = r[active_lb]
    dual[active_ub] = -r[active_ub]
    tmp_index = copy(index)
    tmp_index[active] .= 0
    inactive  = findall(x->x!=0, tmp_index)
    p_inactive = inactive[:]
    chol_A = MatrixFactorizations.cholesky(J[inactive,inactive])

    norm_residual_Ω = norm(reduced_residual(r, u, lb, ub))
    show_trace && print("HIK: Iteration 0, residual norm = $norm_residual_Ω\n")

    while (norm_residual_Ω) > tol && (iter < max_iter)
        
        update = zeros(T, n);
        update[active_lb] = lb[active_lb] - u[active_lb]
        update[active_ub] = ub[active_ub] - u[active_ub]

        if ~isempty(active)
            cr = r[inactive] + J[inactive, active]*update[active]
        else
            cr = r[inactive]
        end

        if p_inactive ≠ inactive
            chol_A = MatrixFactorizations.cholesky(J[inactive,inactive])
        end

        update[inactive] = ldiv!(chol_A, -cr)
        u += damping*update
    
        # which way should the sign be?
        dual[inactive] .= zero(T);
        dual[active_lb] = r[active_lb]
        dual[active_ub] = -r[active_ub]
        
        active_lb = findall((dual .- u .+ lb).>0)
        active_ub = findall((dual .-ub .+u) .>0)
        active = vcat(active_lb, active_ub)

        # print(active)

        tmp_index = copy(index)
        tmp_index[active] .= 0
        p_inactive = inactive[:]
        inactive  = findall(x->x!=0, tmp_index)
        
        project!(u,lb,ub)

        r = J*u - Mf
        norm_residual_Ω = norm(reduced_residual(r, u, lb, ub))


        iter += 1
        show_trace && print("HIK: Iteration $iter, residual norm = $norm_residual_Ω\n")
    end
    
    if iter == max_iter
        @warn("HIK: Iteration max reached.")
    end
    dual[inactive] .= zero(T);
    dual[active_lb] = r[active_lb]
    dual[active_ub] = -r[active_ub]
    return u, dual, iter
end

function evaluate(HIK::HIKSolver{T}, u::AbstractVector{T}, x::AbstractVector{T}) where T
    @assert HIK.d == 1
    Dp = HIK.Dp
    (Dp[:,Block(1)]*u)[x]
end