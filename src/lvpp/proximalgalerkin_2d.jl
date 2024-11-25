struct ObstacleProblem2D{T}
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
    plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    plan_tP::PiecewiseOrthogonalPolynomials.ApplyPlan{T}
    f::AbstractVector{T}
    φ::AbstractVector{T}
    ψ::AbstractVector{T}
    Ux::AbstractArray{T}
    K1::Vector{<:Int}
    K2::Vector{<:Int}
end

function ObstacleProblem2D(r::AbstractVector{T}, p::Integer, f::AbstractVector{T}, φ::AbstractVector{T},
        plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T};matrixfree::Bool=true) where T
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

    Bb = (Dp' * P)[Block.(1:p+1), Block.(1:p)]
    B1 = ExtendableSparseMatrix(size(Bb)...)
    B1 .= Bb
    B1 = sparse(B1)
    B = kron(B1,B1)


    Mp = Diagonal((P' * P)[Block.(1:p), Block.(1:p)])
    M = sparse(kron(Mp,Mp))

    if matrixfree
        E = preconditioner_E(r, p, Nh)
    else
        E = spzeros(1,1)
    end

    D = sparsity_pattern_D(size(B,2), p, Nh, T)
    ψ = NaN*ones(size(B,2))
    Ux,K1,K2 = cache_quadrature2D(Nh, p, lastindex(B,2), plan_tP)


    ObstacleProblem2D{T}(A, chol_A, p, Nh, B, E, D, M, P, Dp, plan_P, plan_tP, f, φ, ψ, Ux, K1, K2)
end

function preconditioner_E3(r::AbstractVector{T}, p::Integer, Nh::Integer) where T
    
    Dp = DirichletPolynomial(r)
    P = ContinuousPolynomial{0}(r)

    Δ = weaklaplacian(Dp)
    ALb = sparse(Diagonal(Symmetric(-parent(Δ)[Block.(1:p+1),Block.(1:p+1)])))
    # MLb = sparse(Symmetric((Dp' * Dp)[Block.(1:p+1),Block.(1:p+1)]))

    M1 = (Dp' * Dp)[Block.(1:p+1),Block.(1:p+1)]
    # DM1 = sparse(Symmetric(BandedMatrix(0=>view(M1, band(0)),Nh=>view(M1, band(Nh)), 2*Nh=>view(M1, band(2*Nh)))))
    DM1 = sparse(Symmetric(BandedMatrix(0=>view(M1, band(0)), 2*Nh=>view(M1, band(2*Nh)))))
  
    # As = sparse(kron(Diagonal(A1),DM1)+kron(DM1,Diagonal(A1)))
    # DB1 = sparse(Matrix(BandedMatrix(-Nh+1=>view(B1,band(-Nh+1)), 0=>view(B1,band(0)), Nh+1=>view(B1,band(Nh+1)))))[:,1:Nh*p]
    # DB = kron(DB1,DB1)

    B1 = sparse(Matrix((Dp' * P)[Block.(1:p+1),Block.(1:p)]))
    # DB1 = sparse(Matrix(BandedMatrix(-Nh+1=>view(B1,band(-Nh+1)), 1=>view(B1,band(1)), Nh+1=>view(B1,band(Nh+1)))))[:,1:Nh*p]
    DB1 = sparse(Matrix(BandedMatrix(-Nh+1=>view(B1,band(-Nh+1)), Nh+1=>view(B1,band(Nh+1)))))[:,1:Nh*p]


    AL2 = kron(ALb, DM1) + kron(DM1, ALb)
    ML4 = kron(DB1,DB1)

    chol_AL2 = MatrixFactorizations.cholesky(AL2)
    sparse(ML4' * ldiv!(chol_AL2, Matrix(ML4)))

end

function preconditioner_E2(r::AbstractVector{T}, p::Integer, Nh::Integer) where T
    
    Dp = DirichletPolynomial(r)
    P = ContinuousPolynomial{0}(r)

    Δ = weaklaplacian(Dp)
    ALb = sparse(Symmetric(-parent(Δ)[Block.(2:p+1),Block.(2:p+1)]))
    MLb = sparse(Symmetric((Dp' * Dp)[Block.(2:p+1),Block.(2:p+1)]))
    ML3b = sparse(Symmetric((P' * Dp)[Block.(1:p),Block.(2:p+1)]))

    AL2 = kron(ALb, MLb) + kron(MLb, ALb) #+  1e-5*kron(MLb, MLb)
    ML4 = kron(ML3b, ML3b)

    chol_AL2 = MatrixFactorizations.cholesky(AL2)
    sparse(ML4 * ldiv!(chol_AL2, Matrix(ML4)'))

end

function preconditioner_E(r::AbstractVector{T}, p::Integer, Nh::Integer) where T
    
    
    Pl = Legendre{T}()
    Ap = sparse(view(diff(Pl)' * diff(Pl), 1:p+2,1:p+2))
    AL = Ap[1:p,1:p] + Ap[3:p+2,3:p+2] - Ap[1:p,3:p+2] - Ap[3:p+2,1:p]
    Mp = sparse(view(Pl'*Pl, 1:p+2,1:p+2))
    ML = Mp[1:p,1:p] + Mp[3:p+2,3:p+2] - Mp[1:p,3:p+2]- Mp[3:p+2,1:p]
    ML3 = Mp[1:p,1:p] - Mp[1:p,3:p+2] 


    s = r[2:end]-r[1:end-1]

    if sum(s .≈ s[1]) == length(s)
        cs = 2 / s[1]
        Nh = 1
    else
        cs = 2 ./ s
    end

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
    AL2 = kron(ALb, MLb) + kron(MLb, ALb)
    ML4 = kron(ML3b, ML3b)
    chol_AL2 = MatrixFactorizations.cholesky(AL2)
    E = sparse(ML4 * (chol_AL2 \ ML4))
    if sum(s .≈ s[1]) == length(s)
        return interlace_blocks(kron(E,Diagonal(ones(length(r)-1))), p, length(r)-1)
    else
        return E
    end
end

function interlace_blocks(A::SparseMatrixCSC{<:T, <:Int}, k::Int, nb::Int) where T
    m = size(A, 1) ÷ k
    n = k * nb * m
    C = ExtendableSparseMatrix(n,n)
    for i in 1:k, j in 1:k
        block_A = A[(i-1)*m+1:i*m, (j-1)*m+1:j*m]
        for l in 1:nb
            C[(nb*i-l)*m+1:(nb*i-(l-1))*m, (nb*j-l)*m+1:(nb*j-(l-1))*m] .= block_A
        end
    end
    return C
end

function cache_quadrature2D(Nh::Int, p::Int, nb::Int, G::PiecewiseOrthogonalPolynomials.ApplyPlan{T}) where T
    cidxs = Vector{Int64}[]
    for j in 1:p
        cidx_ = ((j-1)*Nh + 1):j*Nh
        cidx = copy(cidx_)
        for i in 1:Nh*p
            cidx = cidx ∪ (cidx_.+i*p*Nh)
        end
        for i in 0:p-1
            push!(cidxs, cidx[i*Nh^2+1:(i+1)*Nh^2])
        end
    end

    X = BlockedArray{T}(undef, (BlockedOneTo(Nh:Nh:Nh*p), BlockedOneTo(Nh:Nh:Nh*p), 1:p^2))
    
    for j in 1:p^2
        c = zeros(T,nb)
        c[cidxs[j]] .= one(T)
        X[:,:,j] = c
    end

    K1 = repeat(1:Nh^2*p^2, p^2)
    colM = Matrix{Int}(undef, Nh^2*p^2,p^2)
    for j in 1:p^2                                                                                                                                                                            
        col_idx = Int[]                                                                                                                                                                       
        for k in 1:Nh                                                                                                                                                                         
            col_idx = vcat(col_idx, repeat(cidxs[j][(k-1)*Nh+1:k*Nh], p))                                                                                                                                                                                                                                                              
        end                                                                                                                                                                                   
        colM[:,j] = repeat(col_idx, p)                                                                                                                                                        
    end
    K2 = colM[:]
    G \ X, K1, K2
end

function sparsity_pattern_D(nb::Integer, p::Integer, Nh::Integer, T::Type)
    D = ExtendableSparseMatrix(nb, nb)
    cidxs = Vector{Int64}[]
    for j in 1:p
        cidx_ = ((j-1)*Nh + 1):j*Nh
        cidx = copy(cidx_)
        for i in 1:Nh*p
            cidx = cidx ∪ (cidx_.+i*p*Nh)
        end
        for i in 0:p-1
            push!(cidxs, cidx[i*Nh^2+1:(i+1)*Nh^2])
        end
    end
    for j in 1:p^2
        c = zeros(T,nb)
        c[cidxs[j]] .= one(T)
        col_idx = Int64[]
        for k in 1:Nh
            col_idx = vcat(col_idx, repeat(cidxs[j][(k-1)*Nh+1:k*Nh], p))
        end
        col_idx = repeat(col_idx, p)
        for k in 1:Nh^2*p^2
            D[k,col_idx[k]] = one(T)
        end
    end
    return sparse(D)
end

function ObstacleProblem2D(r::AbstractVector{T}, p::Integer, f::Function, φ::Function;matrixfree::Bool=true) where T
    P = ContinuousPolynomial{0}(r)

    xy, plan_P = grid(P, Block(p,p)), plan_piecewise_legendre_transform(r,(Block(p), Block(p)),(1,2))
    x,y = first(xy), last(xy)

    fv = (plan_P * f.(x,reshape(y,1,1,size(y)...)))[:]
    φv = (plan_P * φ.(x,reshape(y,1,1,size(y)...)))[:]

    ObstacleProblem2D(r, p, fv, φv, plan_P,matrixfree=matrixfree)
end

function ObstacleProblem2D(r::AbstractVector{T}, p::Integer, f::AbstractMatrix{T}, φ::Function;matrixfree::Bool=true, b::Integer=-1) where T
    P = ContinuousPolynomial{0}(r)
    xy, plan_P = grid(P, Block(p,p)), plan_piecewise_legendre_transform(r,(Block(p), Block(p)),(1,2))
    x,y = first(xy), last(xy)
    φv = (plan_P * φ.(x,reshape(y,1,1,size(y)...)))[:]
    ObstacleProblem2D(r, p, f, φv, plan_P,matrixfree=matrixfree)
end

function ObstacleProblem2D(r::AbstractVector{T}, p::Integer, f::Function, φ::AbstractMatrix{T};matrixfree::Bool=true, b::Integer=-1) where T
    P = ContinuousPolynomial{0}(r)
    xy, plan_P = grid(P, Block(p,p)), plan_piecewise_legendre_transform(r,(Block(p), Block(p)),(1,2))
    x,y = first(xy), last(xy)
    fv = (plan_P * f.(x,reshape(y,1,1,size(y)...)))[:]
    ObstacleProblem2D(r, p, fv, φ, plan_P,matrixfree=matrixfree)
end

function show(io::IO, PG::ObstacleProblem2D{T}) where T
    print(io, "ObstacleProblem2D 2D, (Nh, p)=($(PG.Nh), $(PG.p)).")
end