function find_intersect(PG::ObstacleProblem{T}, u::AbstractVector{T}, φ::Function; dom=(0,1), refine_level=3, eps_tol=1) where T
    xx = range(dom...,301)
    Dp = PG.Dp
    valsu = (Dp[:,Block.(1:PG.p+1)]*u)[xx]
    active_set = xx[findall(valsu .≥ (φ.(xx).-eps_tol*eps()))]
    x_ = first(active_set)
    print("x_ is $x_.\n")

    for _ in 2:refine_level
        xx = range(x_ - 2*step(xx), x_ + 2*step(xx), 301)
        valsu = (Dp[:,Block.(1:PG.p+1)]*u)[xx]
        active_set = xx[findall(valsu .≥ (φ.(xx).-eps_tol*eps()))]
        x_ = first(active_set)
        print("x_ is $x_.\n")
    end
    return x_
end

function pg_plotD(PG::ObstacleProblem2D{T}, u::AbstractVector{T};xlim=[0,1]) where T
    nx, px, ny, py = size(PG.plan_P)
    (x,y), plan_D = plan_grid_transform(PG.Dp, Block(PG.p+1, PG.p+1))
    U = BlockMatrix(reshape(u, (nx+1)*px-1, (ny+1)*py-1), [px-1; repeat([px], nx)], [py-1; repeat([py], ny)])
    ux = plan_D \ U
    U = reshape(ux[end:-1:1,:,end:-1:1,:], length(x), length(y))'
    U = hcat(zeros(size(U,1)), U, zeros(size(U,1)))
    U = vcat(zeros(1, size(U,2)), U, zeros(1, size(U,2)))
    vcat(xlim[1], vec(x[end:-1:1,:]), xlim[2]), vcat(xlim[1], vec(y[end:-1:1,:]), xlim[2]),  U
    # surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(ux[end:-1:1,:,end:-1:1,:], length(x), length(y))')
end

function pg_plotP(PG::ObstacleProblem2D{T}, f::AbstractVector{T}) where T
    nx, px, ny, py = size(PG.plan_P)
    (x,y), plan_P = plan_grid_transform(PG.P, Block(PG.p, PG.p))
    F = BlockMatrix(reshape(f, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    fx = plan_P \ F
    F = reshape(fx[end:-1:1,:,end:-1:1,:], length(x), length(y))'
    vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), F
    # surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), reshape(ux[end:-1:1,:,end:-1:1,:], length(x), length(y))')
end

function evaluate2D(u::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, p::Integer, P::PiecewiseOrthogonalPolynomials.AbstractPiecewisePolynomial) where T
    P[y,Block.(1:p)] * reshape(u,isqrt(lastindex(u)), isqrt(lastindex(u)))' * P[x,Block.(1:p)]'   
end

function evaluate2D(u::AbstractVector{T}, x::T, y::T, p::Integer, P::PiecewiseOrthogonalPolynomials.AbstractPiecewisePolynomial) where T
    P[y,Block.(1:p)]' * reshape(u,isqrt(lastindex(u)), isqrt(lastindex(u))) * P[x,Block.(1:p)]
end

function evaluate_u(PG::AdaptiveObstacleProblem{T}, x::AbstractVector{T}, u::AbstractVector{T}) where T
    @assert length(u) == size(PG.A,1)
    ud = pad_u(PG, u)
    (PG.Dp*pad(ud, axes(PG.Dp,2)))[x]
end

function evaluate_u(PG::ObstacleProblem{T}, x::AbstractVector{T}, u::AbstractVector{T}) where T
    @assert length(u) == size(PG.A,1)
    (PG.Dp*pad(u, axes(PG.Dp,2)))[x]
end