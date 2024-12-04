"""
Struct for implementing p-adaptive 1D obstacle problem solver 
with zero bcs. I.e. find u ∈ K satisfying

    (∇u, ∇(v-u)) ≥ (f,v-u) ∀ v ∈ K

where K = {u ∈ H^1_0(Ω) : u ≤ φ}.

The LVPP subproblem is:
    α(∇u, ∇v) + (ψ, v) = α(f, v) + (ψ_, v) ∀ v ∈ H^1_0(Ω)
    (u, q) + (exp(-ψ),q) =0                ∀ q ∈ L^∞(Ω)

"""

struct AdaptiveObstacleProblem{T}
    A::AbstractMatrix{T}
    chol_A
    p::AbstractVector{<:Integer}
    Nh::Integer
    B::AbstractMatrix{T}
    E::AbstractMatrix{T}
    D::AbstractMatrix{T}
    M::AbstractMatrix{T}
    P::ContinuousPolynomial
    Dp::DirichletPolynomial
    plan_Ps#::AbstractVector{<:ApplyPlan{<:T}}
    plan_D::PiecewiseOrthogonalPolynomials.DirichletPolynomialTransform{T}
    f::AbstractVector{T}
    φ::AbstractVector{T}
    ψ::AbstractVector{T}
    idx_M::AbstractVector{<:Integer}
    Ux::Vector{Vector{Matrix{<:T}}}
end


function AdaptiveObstacleProblem(r::AbstractVector{T}, p::AbstractVector{<:Integer}, f::AbstractVector{T}, φ::AbstractVector{T}, idx_M::AbstractVector{<:Integer}; compute_chol=true) where T
    Nh = length(r)-1
    @assert length(p) == Nh

    Dp = DirichletPolynomial(r)
    P = ContinuousPolynomial{0}(r)

    max_p = maximum(p)
    KR = Block.(oneto(max_p+1))
    plan_D = plan_transform(Dp, (Block(max_p+1),), 1)

    Δ = weaklaplacian(Dp)
    idx_A = active_indices_A(p, Nh)
    idx_M = active_indices_M(p, Nh)
    A = sparse(Symmetric(-parent(Δ)[KR,KR]))[idx_A,idx_A]
    if compute_chol == true
        chol_A = MatrixFactorizations.cholesky(A)
    else
        chol_A = MatrixFactorizations.cholesky([1.0 0.0;0.0 1.0])
    end
    
    # B = sparse(view(((P\Dp)'* P' * P),Block.(1:max_p+1), Block.(1:max_p)))[idx_A, idx_M]
    Bb = (Dp' * P)[Block.(1:max_p+1), Block.(1:max_p)]
    B = ExtendableSparseMatrix(size(Bb)...)
    B .= Bb
    B = sparse(B)[idx_A, idx_M]

    # M = ((P' * P)[Block.(1:p), Block.(1:p)])
    M = sparse((P' * P)[Block.(1:max_p), Block.(1:max_p)])[idx_M, idx_M]
    E = sparse(B'*(Diagonal(A)\B))
    up = unique(p)
    plan_Ps = ApplyPlan{<:T}[]
    for ups in up
        np = sum(p .== ups)
        # F = plan_transform(Legendre(), (ups, np,), 1)
        F = plan_legendre_transform(T, (ups, np,), (1,))
        push!(plan_Ps, ApplyPlan(_perm_blockvec, F, (1,)))
    end
   
    # intervals = ClosedInterval.(r[1:end-1],r[2:end])
    # Ps = legendre.(intervals)
    # plan_Ps = plan_grid_transform.(Ps, p)
    D = ExtendableSparseMatrix(Nh*max_p, Nh*max_p)
    ψ = NaN*ones(size(B,2))
    Ux = cache_quadrature_adaptive(Nh, p, plan_Ps, idx_M)
    AdaptiveObstacleProblem{T}(A, chol_A, copy(p), Nh, B, E, D, M, P, Dp, plan_Ps, plan_D, f, φ, ψ, idx_M, Ux)
end

function cache_quadrature_adaptive(Nh::Int, p::AbstractVector{<:Integer}, plan_Ps, idx_M::AbstractVector{<:Integer})
    T = Float64
    max_p = maximum(p)
    up = unique(p)
    Tcx = [Matrix{<:T}[] for _ in 1:max_p]
    for j in 1:max_p
        c = zeros(T, max_p*Nh)
        c[((j-1)*Nh + 1):j*Nh] .= one(T)
        c = c[idx_M]
        
        cs = _cellwise_decomposition(p, Nh, idx_M, c)
        for i in 1:lastindex(plan_Ps)
            ups = up[i]
            plan_P = plan_Ps[i]
    
            idx = findall(p .== ups)
            tcs = cs[idx]
            # tψx[tψx .< -10] .= -10.0
            tcx = plan_P \ BlockVec(reduce(hcat, tcs)')

            push!(Tcx[j], tcx)
        end
    end
    return Tcx
end

function AdaptiveObstacleProblem(r::AbstractVector{T}, p::AbstractVector{<:Integer}, f::Function, φ::Function; compute_chol=true) where T
    intervals = ClosedInterval.(r[1:end-1],r[2:end])
    Ps = legendre.(intervals)
    plan_Ps = plan_grid_transform.(Ps, p)

    Nh = length(r)-1
    max_p = maximum(p)
    fv = zeros(max_p, lastindex(p))
    φv = zeros(max_p, lastindex(p))
    for i in 1:lastindex(p)
        x, plan_P = plan_Ps[i]
        fv[:,i] = pad(Vector(plan_P * f.(x)), 1:max_p)
        φv[:,i] = pad(Vector(plan_P * φ.(x)), 1:max_p)
    end

    
    idx_M = active_indices_M(p, Nh)
    fv = fv'[:][idx_M]
    φv = φv'[:][idx_M]
    # fv = Vector((P \ f.(axes(P,1)))[Block.(1:p)])
    # φv = Vector((P \ φ.(axes(P,1)))[Block.(1:p)])
    AdaptiveObstacleProblem(r, p, fv, φv, idx_M, compute_chol=compute_chol)
end

function active_indices_A(p::AbstractVector{<:Integer}, Nh::Integer)
    @assert lastindex(p) == Nh
    idx = Vector(1:Nh-1)
    for j in 0:Nh-1
        append!(idx, Nh+j:Nh:p[j+1]*Nh+j)
    end
    sort(idx)
end

function active_indices_M(p::AbstractVector{<:Integer}, Nh::Integer)
    @assert lastindex(p) == Nh
    idx = Vector{Integer}()
    for j in 1:Nh
        append!(idx, j:Nh:p[j]*Nh)
    end
    sort(idx)
end

function pad_u(PG::AdaptiveObstacleProblem{T}, u::AbstractVector{T}) where T
    @assert length(u) == size(PG.A,1)

    ud = zeros((maximum(PG.p)+1)*PG.Nh-1)
    idx_A = active_indices_A(PG.p, PG.Nh)
    ud[idx_A] = u
    return ud
end

function pad_ψ(PG::AdaptiveObstacleProblem{T}, ψ::AbstractVector{T}) where T
    @assert length(ψ) == size(PG.B,2)

    ψd = zeros(maximum(PG.p)*PG.Nh)
    idx_M = active_indices_M(PG.p, PG.Nh)
    ψd[idx_M] = ψ
    return ψd
end

function _cellwise_decomposition(p::AbstractVector{<:Integer}, Nh::Integer, idx_M::AbstractVector{<:Integer}, u::AbstractVector{T}) where T
    max_p = maximum(p)
    fu = zeros(Nh*max_p)
    fu[idx_M] = u

    us = AbstractVector{T}[]
    for i in 1:Nh
        append!(us, [fu[i:Nh:p[i]*Nh]])
    end
    us
end

function cellwise_decomposition(PG::AdaptiveObstacleProblem, u::AbstractVector{T}) where T
    _cellwise_decomposition(PG.p, PG.Nh, PG.idx_M, u)
end

function cellwise_interlace(PG::AdaptiveObstacleProblem, u::AbstractVector{<:AbstractVector{<:T}}) where T
    p, idx_M = PG.p, PG.idx_M
    
    max_p = maximum(p)

    uv = zeros(max_p, lastindex(p))
    for i in 1:lastindex(p)
        uv[:,i] = pad(u[i], 1:max_p)
    end

    uv'[:][idx_M]

end

function show(io::IO, PG::AdaptiveObstacleProblem{T}) where T
    print(io, "AdaptiveObstacleProblem, (Nh, p)=($(PG.Nh), $(PG.p)).")
end