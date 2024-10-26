struct ObstacleProblem{T}
    A::AbstractMatrix{T}
    chol_A#::MatrixFactorizations.ReverseCholesky{T}
    p::Integer
    Nh::Integer
    B::AbstractMatrix{T}
    E::AbstractMatrix{T}
    D::AbstractMatrix{T}
    M::AbstractMatrix{T}
    P::ContinuousPolynomial
    Dp::DirichletPolynomial
    plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    plan_tP::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    f::AbstractVector{T}
    φ::AbstractVector{T}
    ψ::AbstractVector{T}
    Ux::AbstractArray{T}
    K1::Vector{<:Int}
    K2::Vector{<:Int}
end

function ObstacleProblem(r::AbstractVector{T}, p::Integer, f::AbstractVector{T}, φ::AbstractVector{T}; b::Integer=-1, compute_rchol=true) where T
    Dp = DirichletPolynomial(r)
    P = ContinuousPolynomial{0}(r)
    Nh = length(r)-1

    KR = Block.(oneto(p+1))

    Δ = weaklaplacian(Dp)
    A = (Symmetric(-parent(Δ)[KR,KR]))
    A = sparse(A)
    # if compute_rchol == true
    #     rchol_A = MatrixFactorizations.reversecholesky(A)
    # else
    #     rchol_A = MatrixFactorizations.reversecholesky([1.0 0.0;0.0 1.0])
    # end

    
    if compute_rchol == true
        rchol_A = MatrixFactorizations.cholesky(A)
    else
        rchol_A = MatrixFactorizations.cholesky([1.0 0.0;0.0 1.0])
    end
    
    Bb = (Dp' * P)[Block.(1:p+1), Block.(1:p)]
    B = ExtendableSparseMatrix(size(Bb)...)
    B .= Bb
    B = sparse(B)



    M = Diagonal((P' * P)[Block.(1:p), Block.(1:p)])

    # E = sparse(B'*(Diagonal(A)\B))
    E = preconditioner_E_1D(r, p, Nh)


    plan_P = plan_piecewise_legendre_transform(r, (Block(p),), 1)
    plan_tP = plan_piecewise_legendre_transform(r, (Block(p),p), 1)

    D = ExtendableSparseMatrix(Nh*p, Nh*p)
    ψ = NaN*ones(size(B,2))

    Ux,K1,K2 = cache_quadrature(Nh, p, lastindex(B,2), plan_tP)

    ObstacleProblem{T}(A, rchol_A, p, Nh, B, E, D, M, P, Dp, plan_P, plan_tP, f, φ, ψ, Ux, K1, K2)
end

function cache_quadrature(Nh::Int, p::Int, nb::Int, G::PiecewiseOrthogonalPolynomials.ApplyPlan{T}) where T
    cidxs = Vector{Int64}[]
    for j in 1:p
        push!(cidxs, ((j-1)*Nh + 1):j*Nh)
    end

    X = BlockedMatrix{T}(undef, (BlockArrays.BlockedOneTo(Nh:Nh:Nh*p),1:p))
    for j in 1:p
        c = zeros(T,nb)
        c[cidxs[j]] .= one(T)
        X[:,j] .= c
    end

    colM = Matrix{Int}(undef, Nh*p,p)
    for j in 1:p                                                                                                                                                                           
        colM[:,j] .= repeat((j-1)*Nh+1:j*Nh, p)                                                                                                                                              
    end

    K1 = repeat(1:Nh*p, p)
    K2 = colM[:]

    G \ X, K1, K2
end

function preconditioner_E_1D(r::AbstractVector{T}, p::Integer, Nh::Integer) where T
    
    s = r[2:end]-r[1:end-1]
    cs = 2 ./ s

    Pl = Legendre{T}()
    Ap = sparse(view(diff(Pl)' * diff(Pl), 1:p+2,1:p+2))
    AL = Ap[1:p,1:p] + Ap[3:p+2,3:p+2] - Ap[1:p,3:p+2] - Ap[3:p+2,1:p]
    Mp = sparse(view(Pl'*Pl, 1:p+2,1:p+2))
    ML = Mp[1:p,1:p] + Mp[3:p+2,3:p+2] - Mp[1:p,3:p+2]- Mp[3:p+2,1:p]
    ML3 = Mp[1:p,1:p] - Mp[1:p,3:p+2] 

    ALb = ExtendableSparseMatrix(p*Nh,p*Nh)
    MLb = ExtendableSparseMatrix(p*Nh,p*Nh)
    ML3b = ExtendableSparseMatrix(p*Nh,p*Nh)
    for i in axes(AL,1), j in axes(AL,2)
        for k in 0:Nh-1
            ALb[Nh*i-k,Nh*j-k] = cs[k+1] * AL[i,j]
            MLb[Nh*i-k,Nh*j-k] = ML[i,j] / cs[k+1]
            ML3b[Nh*i-k,Nh*j-k] = ML3[i,j] / cs[k+1]
        end
    end

    chol_ALb = MatrixFactorizations.cholesky(ALb)
    sparse(ML3b * ldiv!(chol_ALb, Matrix(ML3b)'))
end

function ObstacleProblem(r::AbstractVector{T}, p::Integer, f::Function, φ::Function;compute_rchol=true) where T
    P = ContinuousPolynomial{0}(r)
    # x, plan_P = plan_grid_transform(P, (Block(p),), 1)
    x, plan_P = grid(P, Block(p)), plan_piecewise_legendre_transform(r, (Block(p),), 1)
    fv = Vector(plan_P * f.(x))
    φv = Vector(plan_P * φ.(x))

    ObstacleProblem(r, p, fv, φv, compute_rchol=compute_rchol)
end

function ObstacleProblem(r::AbstractVector{T}, p::Integer, f::AbstractVector{T}, φ::Function;compute_rchol=true) where T
    P = ContinuousPolynomial{0}(r)
    # x, plan_P = plan_grid_transform(P, (Block(p),), 1)
    x, plan_P = grid(P, Block(p)), plan_piecewise_legendre_transform(r, (Block(p),), 1)
    φv = plan_P * φ.(x)

    ObstacleProblem(r, p, f, φv, compute_rchol=compute_rchol)
end

function ObstacleProblem(r::AbstractVector{T}, p::Integer, f::Function, φ::AbstractVector{T};compute_rchol=true) where T
    P = ContinuousPolynomial{0}(r)
    # x, plan_P = plan_grid_transform(P, (Block(p),), 1)
    x, plan_P = grid(P, Block(p)), plan_piecewise_legendre_transform(r, (Block(p),), 1)
    fv = plan_P * f.(x)
    ObstacleProblem(r, p, fv, φ, compute_rchol=compute_rchol)
end

function show(io::IO, PG::ObstacleProblem{T}) where T
    print(io, "ObstacleProblem, (Nh, p)=($(PG.Nh), $(PG.p)).")
end