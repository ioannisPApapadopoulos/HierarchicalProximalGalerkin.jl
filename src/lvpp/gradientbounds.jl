struct GradientBounds{T}
    A::AbstractMatrix{T}
    chol_A
    p::Integer
    Nh::Integer
    B::AbstractMatrix{T}
    E::AbstractMatrix{T}
    D::AbstractMatrix{T}
    M::AbstractMatrix{T}
    G::AbstractMatrix{T}
    P::ContinuousPolynomial
    Dp::DirichletPolynomial
    plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    plan_dP::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    plan_tP::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    f::AbstractVector{T}
    φ::AbstractVector{T}
    φx::AbstractArray{T}
    ψ::AbstractVector{T}
    Ux::AbstractArray{T}
    K1::Vector{<:Int}
    K2::Vector{<:Int}
end

function GradientBounds(r::AbstractVector{T}, p::Integer, f::AbstractVector{T}, φ::AbstractVector{T},
        plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T}) where T
    Dp = DirichletPolynomial(r)
    P = ContinuousPolynomial{0}(r)
    Nh = length(r)-1

    KR = Block.(oneto(p))
    plan_tP = plan_piecewise_legendre_transform(r,(Block(p), 2*p^2),(1,))
    plan_dP = plan_piecewise_legendre_transform(r,(Block(p),  2), (1,))

    Δ = weaklaplacian(Dp)
    A = sparse(Symmetric(-parent(Δ)[KR,KR]))
    chol_A = MatrixFactorizations.cholesky(A)

    Bb = (diff(Dp)' * P)[Block.(1:p), Block.(1:p)]
    B = ExtendableSparseMatrix(size(Bb)...)
    B .= Bb
    B = sparse(B)

    Gb = (Dp' * P)[Block.(1:p), Block.(1:p)]
    G = ExtendableSparseMatrix(size(Gb)...)
    G .= Gb
    G = sparse(G)

    M = sparse(Diagonal((P' * P)[Block.(1:p), Block.(1:p)]))

    E = zeros(1,1) #preconditioner_E_gradient_bounds(r, p, Nh)

    D = ExtendableSparseMatrix(size(B,2), size(B,2))
    ψ = NaN*ones(size(B,2))
    Ux,K1,K2 = cache_quadrature2D(Nh, p, lastindex(B,2)÷2, plan_piecewise_legendre_transform(r, (Block(p),p), 1))

    φx = plan_P \ BlockVec(reshape(copy(φ),reverse(size(plan_P))))
    
    GradientBounds{T}(A, chol_A, p, Nh, B, E, D, M, G, P, Dp, plan_P, plan_dP, plan_tP, f, φ, φx, ψ, Ux, K1, K2)
end

# function preconditioner_E_gradient_bounds(r::AbstractVector{T}, p::Integer, Nh::Integer) where T
    
#     s = r[2:end]-r[1:end-1]
#     cs = 2 ./ s
#     Pl = Legendre{T}()
#     Ap = sparse(view(diff(Pl)' * diff(Pl), 1:p+2,1:p+2))
#     AL = Ap[1:p,1:p] + Ap[3:p+2,3:p+2] - Ap[1:p,3:p+2] - Ap[3:p+2,1:p]
#     Mp = sparse(view(Pl'*Pl, 1:p+2,1:p+2))
#     ML = Mp[1:p,1:p] + Mp[3:p+2,3:p+2] - Mp[1:p,3:p+2]- Mp[3:p+2,1:p]

#     # Bp = sparse(view(diff(Pl)'*Pl, 1:p+3,1:p+3))
#     # BL3 = Bp[1:p,1:p] + Bp[3:p+2,3:p+2] - Bp[3:p+2,1:p] - Bp[1:p,3:p+2]
#     # Bp = sparse(view(Pl'*diff(Pl), 1:p+3,1:p+3))
#     # ML3 = Bp[1:p,1:p] + Bp[3:p+2,3:p+2] - Bp[3:p+2,1:p] - Bp[1:p,3:p+2]

#     ALb = ExtendableSparseMatrix(p*Nh,p*Nh)
#     MLb = ExtendableSparseMatrix(p*Nh,p*Nh)
#     # BL3b = ExtendableSparseMatrix(p*Nh,p*Nh)
#     # ML3b = ExtendableSparseMatrix(p*Nh,p*Nh)
#     for i in axes(AL,1), j in axes(AL,2)
#         for k in 0:Nh-1
#             ALb[Nh*i-k,Nh*j-k] = cs[k+1] * AL[i,j]
#             MLb[Nh*i-k,Nh*j-k] = ML[i,j] / cs[k+1]
#             # BL3b[Nh*i-k,Nh*j-k] = BL3[i,j] #/ (cs[k+1])
#             # ML3b[Nh*i-k,Nh*j-k] = ML3[i,j] / cs[k+1]
#         end
#     end
#     AL2 = kron(ALb, MLb) + kron(MLb, ALb) #+ kron(BL3b, BL3b) + kron(ML3b, ML3b)
#     return sparse(AL2)
# end

function GradientBounds(r::AbstractVector{T}, p::Integer, f::Function, φ::Function) where T
    P = ContinuousPolynomial{0}(r)

    x, plan_P = grid(P, Block(p)), plan_piecewise_legendre_transform(r, (Block(p),), 1)
    fv = Vector(plan_P * f.(x))
    φv = Vector(plan_P * φ.(x))
    GradientBounds(r, p, fv, φv, plan_P)
end

function GradientBounds(r::AbstractVector{T}, p::Integer, f::AbstractMatrix{T}, φ::Function; b::Integer=-1) where T
    P = ContinuousPolynomial{0}(r)
    x, plan_P = grid(P, Block(p)), plan_piecewise_legendre_transform(r, (Block(p),), 1)
    φv = plan_P * φ.(x)
    GradientBounds(r, p, f, φv, plan_P)
end

function GradientBounds(r::AbstractVector{T}, p::Integer, f::Function, φ::AbstractMatrix{T}; b::Integer=-1) where T
    P = ContinuousPolynomial{0}(r)
    x, plan_P = grid(P, Block(p)), plan_piecewise_legendre_transform(r, (Block(p),), 1)
    fv = plan_P * f.(x)
    GradientBounds(r, p, fv, φ, plan_P)
end

function show(io::IO, PG::GradientBounds{T}) where T
    print(io, "GradientBounds 1D, (Nh, p)=($(PG.Nh), $(PG.p)).")
end