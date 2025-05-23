"""
Struct for implementing 2D gradient bounds solver with
zero bcs. I.e. find u ∈ K satisfying

    (∇u, ∇(v-u)) ≥ (f,v-u) ∀ v ∈ K

where K = {u ∈ H^1_0(Ω) : |∇u| ≤ φ}.

The LVPP subproblem is:
    α(∇u, ∇v) + (ψ, ∇v) = α(f, v) + (ψ_, ∇v) ∀ v ∈ H^1_0(Ω)
    (∇u, q) - (φψ/sqrt(1+|ψ|^2),q) =0        ∀ q ∈ L^∞(Ω)^2

"""

struct GradientBounds2D{T}
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

function GradientBounds2D(r::AbstractVector{T}, p::Integer, f::AbstractVector{T}, φ::AbstractVector{T},
        plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T}) where T
    Dp = DirichletPolynomial(r)
    P = ContinuousPolynomial{0}(r)
    Nh = length(r)-1

    KR = Block.(oneto(p))
    plan_tP = plan_piecewise_legendre_transform(r,(Block(p), Block(p), 2*p^2),(1,2))
    plan_dP = plan_piecewise_legendre_transform(r,(Block(p), Block(p), 2),(1,2))

    Δ = weaklaplacian(Dp)
    A1 = sparse(Symmetric(-parent(Δ)[KR,KR]))
    M1 = sparse(Symmetric((Dp' * Dp)[KR,KR]))
    A = sparse(kron(A1,M1) + kron(M1,A1))
    chol_A = MatrixFactorizations.cholesky(A)

    Bb = (diff(Dp)' * P)[Block.(1:p), Block.(1:p)]
    B1 = ExtendableSparseMatrix(size(Bb)...)
    B1 .= Bb
    B1 = sparse(B1)

    Gb = (Dp' * P)[Block.(1:p), Block.(1:p)]
    G1 = ExtendableSparseMatrix(size(Gb)...)
    G1 .= Gb
    G1 = sparse(G1)

    B = [kron(B1,G1) kron(G1,B1)]
    G = kron(G1,G1)

    Mp = Diagonal((P' * P)[Block.(1:p), Block.(1:p)])
    M = sparse(kron(Mp,Mp))

    E = preconditioner_E_gradient_bounds(r, p, Nh)
    E = blockdiag(E,E)

    D = sparsity_pattern_D(size(B,2)÷2, p, Nh, T)
    D = blockdiag(D, D)
    ψ = NaN*ones(size(B,2))
    Ux,K1,K2 = cache_quadrature2D(Nh, p, lastindex(B,2)÷2, plan_piecewise_legendre_transform(r,(Block(p), Block(p), p^2),(1,2)))

    nx, px, ny, py = size(plan_P)
    φx = plan_P \ BlockMatrix(reshape(φ, nx*px, ny*py), repeat([px], nx), repeat([py], ny))

    GradientBounds2D{T}(A, chol_A, p, Nh, B, E, D, M, G, P, Dp, plan_P, plan_dP, plan_tP, f, φ, φx, ψ, Ux, K1, K2)
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
# #     # ML4 = sparse(kron(BL3b, ML3b) + kron(ML3b, BL3b))
# #     # ML4 = [ML4 ML4]
# #     ML4 = [kron(BL3b, ML3b) kron(ML3b, BL3b)]
# #     # ML4 = ML4[:,(p-1)*(Nh)^2+Nh+1:end]

# #     chol_AL2 = MatrixFactorizations.cholesky(AL2)
# #     sparse(ML4' * ldiv!(chol_AL2, Matrix(ML4)))
# end

function preconditioner_E_gradient_bounds(r::AbstractVector{T}, p::Integer, Nh::Integer) where T
    
    s = r[2:end]-r[1:end-1]
    cs = 2 ./ s
    Pl = Legendre{T}()
    Ap = sparse(view(diff(Pl)' * diff(Pl), 1:p+2,1:p+2))
    AL = Ap[1:p,1:p] + Ap[3:p+2,3:p+2] - Ap[1:p,3:p+2] - Ap[3:p+2,1:p]
    Mp = sparse(view(Pl'*Pl, 1:p+2,1:p+2))
    ML = Mp[1:p,1:p] + Mp[3:p+2,3:p+2] - Mp[1:p,3:p+2]- Mp[3:p+2,1:p]

    ALb = ExtendableSparseMatrix(p*Nh,p*Nh)
    MLb = ExtendableSparseMatrix(p*Nh,p*Nh)
    for i in axes(AL,1), j in axes(AL,2)
        for k in 0:Nh-1
            ALb[Nh*i-k,Nh*j-k] = cs[k+1] * AL[i,j]
            MLb[Nh*i-k,Nh*j-k] = ML[i,j] / cs[k+1]
        end
    end
    AL2 = kron(ALb, MLb) + kron(MLb, ALb)
    return sparse(AL2)
end

function GradientBounds2D(r::AbstractVector{T}, p::Integer, f::Function, φ::Function) where T
    P = ContinuousPolynomial{0}(r)

    xy, plan_P = grid(P, Block(p,p)), plan_piecewise_legendre_transform(r,(Block(p), Block(p)),(1,2))
    x,y = first(xy), last(xy)

    fv = (plan_P * f.(x,reshape(y,1,1,size(y)...)))[:]
    φv = (plan_P * φ.(x,reshape(y,1,1,size(y)...)))[:]

    GradientBounds2D(r, p, fv, φv, plan_P)
end

function GradientBounds2D(r::AbstractVector{T}, p::Integer, f::AbstractMatrix{T}, φ::Function; b::Integer=-1) where T
    P = ContinuousPolynomial{0}(r)
    xy, plan_P = grid(P, Block(p,p)), plan_piecewise_legendre_transform(r,(Block(p), Block(p)),(1,2))
    x,y = first(xy), last(xy)
    φv = (plan_P * φ.(x,reshape(y,1,1,size(y)...)))[:]
    GradientBounds2D(r, p, f, φv, plan_P)
end

function GradientBounds2D(r::AbstractVector{T}, p::Integer, f::Function, φ::AbstractMatrix{T}; b::Integer=-1) where T
    P = ContinuousPolynomial{0}(r)
    xy, plan_P = grid(P, Block(p,p)), plan_piecewise_legendre_transform(r,(Block(p), Block(p)),(1,2))
    x,y = first(xy), last(xy)
    fv = (plan_P * f.(x,reshape(y,1,1,size(y)...)))[:]
    GradientBounds2D(r, p, fv, φ, plan_P)
end

function show(io::IO, PG::GradientBounds2D{T}) where T
    print(io, "GradientBounds2D 2D, (Nh, p)=($(PG.Nh), $(PG.p)).")
end