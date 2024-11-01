struct BCsObstacleProblem2D{T}
    A::AbstractMatrix{T}
    chol_A
    p::Integer
    Nh::Integer
    B::AbstractMatrix{T}
    E::AbstractMatrix{T}
    D::AbstractMatrix{T}
    M::AbstractMatrix{T}
    P::ContinuousPolynomial
    Dp::DirichletPolynomial
    C::ContinuousPolynomial
    plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    plan_tP::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    f::AbstractVector{T}
    φ::AbstractVector{T}
    ψ::AbstractVector{T}
    bcs_vals::AbstractVector{T}
    bcs_Fx::AbstractVector{T}
    bcs_Fy::AbstractVector{T}
    bcs_idx::AbstractVector{Integer}
    free_idx::AbstractVector{Integer}
    n2du::Integer
    Ux::AbstractArray{T}
    K1::Vector{<:Int}
    K2::Vector{<:Int}
end

function BCsObstacleProblem2D(r::AbstractVector{T}, p::Integer, f::AbstractVector{T}, φ::AbstractVector{T}, g::AbstractVector{T},
        plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T}) where T
    Dp = DirichletPolynomial(r)
    P = ContinuousPolynomial{0}(r)
    Nh = length(r)-1

    KR = Block.(oneto(p+1))

    plan_tP = plan_piecewise_legendre_transform(r,(Block(p), Block(p), p^2),(1,2))

    Δ = weaklaplacian(Dp)
    A1 = sparse(Symmetric(-parent(Δ)[KR,KR]))
    M1 = sparse(Symmetric((Dp' * Dp)[KR,KR]))
    A = sparse(kron(A1,M1) + kron(M1,A1))


    chol_A = MatrixFactorizations.cholesky(A)

    # B1 = sparse(view(((P\Dp)'* P' * P),Block.(1:p+1), Block.(1:p)))
    Bb = (Dp' * P)[Block.(1:p+1), Block.(1:p)]
    B1 = ExtendableSparseMatrix(size(Bb)...)
    B1[:,:] = Bb
    B1 = sparse(B1)
    B = kron(B1,B1)

    # B = ((P\D)'* P' * P)[KR,KR]
    # M = ((P' * P)[Block.(1:p), Block.(1:p)])
    M = Diagonal((P' * P)[Block.(1:p), Block.(1:p)])
    M = sparse(kron(M,M))

    # E = sparse(B'*(Diagonal(A)\B))
    E = preconditioner_E(r, p, Nh)

    D = sparsity_pattern_D(size(B,2), p, Nh, T)
    

    C = ContinuousPolynomial{1}(r)
    Δ = weaklaplacian(C)
    Ac1 = sparse(Symmetric(-parent(Δ)[KR,KR]))
    Mc1 = sparse(Symmetric((C' * C)[KR,KR]))
    Ac = (kron(Ac1,Mc1) + kron(Mc1,Ac1))
    Ac1[1,1]=NaN
    Ac1[Nh+1, Nh+1] = NaN
    AcNaN = (kron(Ac1,Mc1) + kron(Mc1,Ac1))
    idx = findall(isnan.(AcNaN))
    bcs_idx = unique([idx[i][2] for i in 1:lastindex(idx)])
    free_idx = filter!(e->e∉bcs_idx,Vector(axes(Ac,1)))
    bc_vals = zeros(size(Ac,1))
    bc_vals[bcs_idx] = g[bcs_idx]
    bcs_Fx = (Ac * bc_vals)[free_idx]

    # B1c = sparse(view(((P\C)'* P' * P),Block.(1:p+1), Block.(1:p)))
    Bb = (C' * P)[Block.(1:p+1), Block.(1:p)]
    B1c = ExtendableSparseMatrix(size(Bb)...)
    B1c[:,:] = Bb
    B1c = sparse(B1c)
    Bc = kron(B1c,B1c)

    bcs_Fy = (-Bc' * bc_vals)
    n2du = size(Ac,1)

    ψ = NaN*ones(size(B,2))
    Ux,K1,K2 = cache_quadrature2D(Nh, p, lastindex(B,2), plan_tP)

    BCsObstacleProblem2D{T}(A, chol_A, p, Nh, B, E, D, M, P, Dp, C, plan_P, plan_tP, f, φ, ψ, bc_vals[bcs_idx], bcs_Fx, bcs_Fy, bcs_idx, free_idx, n2du, Ux, K1, K2)
end

function BCsObstacleProblem2D(r::AbstractVector{T}, p::Integer, f::Function, φ::Function, g::Function) where T
    P = ContinuousPolynomial{0}(r)

    xy, plan_P = grid(P, Block(p,p)), plan_piecewise_legendre_transform(r,(Block(p), Block(p)),(1,2))
   
    x,y = first(xy), last(xy)

    fv = (plan_P * f.(x,reshape(y,1,1,size(y)...)))[:]
    φv = (plan_P * φ.(x,reshape(y,1,1,size(y)...)))[:]

    C = ContinuousPolynomial{1}(r)
    xyC, plan_C = plan_grid_transform(C, Block(p+1,p+1))
    x,y = first(xyC), last(xyC)
    gv = (plan_C * g.(x,reshape(y,1,1,size(y)...)))[:]

    BCsObstacleProblem2D(r, p, fv, φv, gv, plan_P)
end

function show(io::IO, PG::BCsObstacleProblem2D{T}) where T
    print(io, "BCsObstacleProblem2D 2D, (Nh, p)=($(PG.Nh), $(PG.p)).")
end