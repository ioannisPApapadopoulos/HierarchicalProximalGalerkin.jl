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
        # da = apply_D_adaptive(PG, c, ψ)
        da = _assemble_D_adaptive(PG, ψ, j)
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

function _assemble_D_adaptive(PG::AdaptiveObstacleProblem{T}, ψ::AbstractVector{T}, j::Int) where T
    M, plan_Ps, p = PG.M, PG.plan_Ps, PG.p
    up = unique(p)

    ψs = cellwise_decomposition(PG, ψ)
    for i in 1:lastindex(plan_Ps)
        ups = up[i]
        plan_P = plan_Ps[i]

        idx = findall(p .== ups)
        tψs = ψs[idx]

        tψx = plan_P \ BlockVec(reduce(hcat, tψs)')
        vals = exp.(-tψx) .* PG.Ux[j][i]
        ψc = plan_P * vals
        rψc = reshape(ψc, reverse(size(plan_P)))
        for j in 1:size(rψc,1) ψs[idx[j]] = rψc[j,:] end
    end
    M * cellwise_interlace(PG, ψs)
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
    plan_dP = PG.plan_dP
    plan_P = PG.plan_P
    nx, px, ny, py = size(plan_P)
    Nh, p = PG.Nh, PG.p

    ψ1, ψ2 = ψ[1:length(ψ)÷2], ψ[length(ψ)÷2+1:end]
    X = BlockedArray{T}(undef, (BlockedOneTo(Nh:Nh:Nh*p), BlockedOneTo(Nh:Nh:Nh*p), 1:2))
    X[:,:,1] .= BlockMatrix(reshape(ψ1, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    X[:,:,2] .= BlockMatrix(reshape(ψ2, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    ψx = plan_dP \ X
    vals1 = (PG.φx .* (one(T) .+ ψx[:,:,:,:,2].^2) ./ (sqrt.(one(T) .+ ψx[:,:,:,:,1].^2 + ψx[:,:,:,:,2].^2)).^3) .* PG.Ux
    vals2 = (PG.φx .* (one(T) .+ ψx[:,:,:,:,1].^2) ./ (sqrt.(one(T) .+ ψx[:,:,:,:,1].^2 + ψx[:,:,:,:,2].^2)).^3) .* PG.Ux
    
    X = Array{T}(PG.plan_tP * [vals1;;;;;vals2])
    
    da1 = PG.M * reshape(X[:,:,1:p^2], (Nh*p)^2, p^2)
    da2 = PG.M * reshape(X[:,:,p^2+1:end], (Nh*p)^2, p^2)
    D1 = sparse(PG.K1, PG.K2, da1[:])
    D2 = sparse(PG.K1, PG.K2, da2[:])
    blockdiag(D1,D2)
end