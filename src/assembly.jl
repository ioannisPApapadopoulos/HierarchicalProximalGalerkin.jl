## ObstacleProblem

function assemble_D(PG::ObstacleProblem{T}, ψ::AbstractVector{T}) where T
    plan_P = PG.plan_P
    ψx = plan_P \ BlockVec(reshape(copy(ψ),reverse(size(plan_P))))
    vals = exp.(-ψx) .* PG.Ux
    X = Array{T}(PG.plan_tP * vals)
    da = PG.M * X
    sparse(PG.K1, PG.K2, da[:])
end

function assemble_D(PG::AdaptiveObstacleProblem{T}, ψ::AbstractVector{T}) where T
    n = size(PG.B,2)
    p, Nh, idx_M = PG.p, PG.Nh, PG.idx_M
    max_p = maximum(p)

   
    das = Vector{T}[]
    for j in 1:max_p
        c = zeros(max_p*Nh)
        c[((j-1)*Nh + 1):j*Nh] .= one(T)
        c = c[idx_M]
        da = apply_D_adaptive(PG, c, ψ)
        push!(das, da)
    end

    D = PG.D
    idx = zeros(Integer, max_p*Nh)
    idx[idx_M] = 1:n
    for i in 1:max_p
        for j in 1:Nh
            r = das[i][idx[j:Nh:Nh*p[j]]]
            D[j:Nh:Nh*p[j], j+Nh*(i-1)] = r
        end
    end
    return sparse(D[idx_M, idx_M])
end

function assemble_D(PG::Union{<:ObstacleProblem2D{T},<:BCsObstacleProblem2D{T}}, ψ::AbstractVector{T}) where T
    B = PG.B
    Nh, p = PG.Nh, PG.p
    plan_P = PG.plan_P
    nx, px, ny, py = size(plan_P)

    ψx = plan_P \ BlockMatrix(reshape(ψ, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    vals = exp.(-ψx) .* PG.Ux
    X = Array{T}(PG.plan_tP * vals)

    da = PG.M * reshape(X, (Nh*p)^2, p^2)
    sparse(PG.K1, PG.K2, da[:])
end


## GradientBounds

function assemble_D(PG::GradientBounds2D{T}, ψ::AbstractVector{T}) where T
    B = PG.B
    Nh, p = PG.Nh, PG.p
    plan_P = PG.plan_P
    nx, px, ny, py = size(plan_P)

    ψx = plan_P \ BlockMatrix(reshape(ψ, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    vals = (PG.φx ./ (sqrt.(one(T) .+ ψx.^2)).^3) .* PG.Ux

    X = Array{T}(PG.plan_tP * vals)

    da = PG.M * reshape(X, (Nh*p)^2, p^2)
    sparse(PG.K1, PG.K2, da[:])
end