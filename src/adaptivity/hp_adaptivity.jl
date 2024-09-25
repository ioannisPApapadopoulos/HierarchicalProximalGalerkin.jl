struct MeshAdaptivity{T}
    PG::Union{<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}, <:HIKSolver{T}}
    α::T
    up::AbstractVector{T}
    exp_ψP::AbstractVector{T}
    λ::AbstractVector{T}
    oc::AbstractVector{T}
    fP::AbstractVector{T}
    φP::AbstractVector{T}
    Rp::AbstractMatrix{T}
    ΔP::AbstractMatrix{T}
    M::AbstractMatrix{T}
end

function MeshAdaptivity(PG::ObstacleProblem{T}, α::T, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, f::Function, φ::Function) where T
    Dp, P, p = PG.Dp, PG.P, PG.p

    λ = (w-ψ)/α
    Rp = (P \ Dp)[Block.(1:p+2), Block.(1:p+1)]
    up = Rp*u

    plan_P = plan_transform(PG.P, (Block(p+2),), 1)
    oc = plan_P * max.(plan_P \ BlockVec(reshape(Vector(copy(up - pad(PG.φ, axes(up,1)))),reverse(size(plan_P)))), 0)

    L = Legendre()
    x = axes(L,1)
    Dm = (Jacobi(2,2) \ (Derivative(x) * (Derivative(x) * L)))[1:p+2,1:p+2]
    Rm = (L \ Jacobi(2,2))[1:p+2,1:p+2]
    ΔP = Rm*Dm

    M = (PG.P' * PG.P)

    # fP = PG.P \ f.(axes(PG.P,1))
    # φP = PG.P \ φ.(axes(PG.P,1))

    x, plan_P = plan_grid_transform(PG.P, (Block(p+12),), 1)
    ψx = plan_P \ BlockVec(reshape(copy(Vector(pad(ψ, PG.Nh*(p+12)))),reverse(size(plan_P))))
    exp_ψP = plan_P * exp.(-ψx)

    fP = plan_P * f.(x)
    φP = plan_P * φ.(x)

    MeshAdaptivity{T}(PG, α, up, exp_ψP, λ, oc, fP, φP, Rp, ΔP, M)
end

function MeshAdaptivity(PG::AdaptiveObstacleProblem{T}, α::T, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, f::Function, φ::Function) where T
    Dp, P, p = PG.Dp, PG.P, PG.p

    λ = (w-ψ)/α
    max_p = maximum(p)
    Rp = (P \ Dp)[Block.(1:max_p+2), Block.(1:max_p+1)]
    ud = pad_u(PG, u)
    up = Rp*ud

    plan_P = plan_transform(PG.P, (Block(max_p+2),), 1)
    oc = plan_P * max.(plan_P \ BlockVec(reshape(Vector(copy(up - pad(pad_ψ(PG,PG.φ),axes(up,1)))),reverse(size(plan_P)))), 0)

    ψx = plan_P \ BlockVec(reshape(copy(Vector(pad(pad_ψ(PG,ψ), axes(up,1)))),reverse(size(plan_P))))
    ψx[ψx.<-100] .= 100
    exp_ψP = plan_P * exp.(-ψx)


    # oc = pad(pad_ψ(PG,PG.φ),axes(up,1)) - up
    L = Legendre()
    x = axes(L,1)
    Dm = (Jacobi(2,2) \ (Derivative(x) * (Derivative(x) * L)))[1:max_p+2,1:max_p+2]
    Rm = (L \ Jacobi(2,2))[1:max_p+2,1:max_p+2]
    ΔP = Rm*Dm

    M = (PG.P' * PG.P)

    # fP = PG.P \ f.(axes(PG.P,1))
    # φP = PG.P \ φ.(axes(PG.P,1))

    x, plan_P = plan_grid_transform(PG.P, (Block(max_p+12),), 1)
    # ψx = plan_P \ BlockVec(reshape(copy(Vector(pad(pad_ψ(PG,ψ), PG.Nh*(max_p+12)))),reverse(size(plan_P))))
    # exp_ψP = plan_P * exp.(-ψx)


    fP = plan_P * f.(x)
    φP = plan_P * φ.(x)

    MeshAdaptivity{T}(PG, α, up, exp_ψP, λ, oc, fP, φP, Rp, ΔP, M)
end

function MeshAdaptivity(HIK::HIKSolver{T}, u::AbstractVector{T}, λ::AbstractVector{T}, f::Function, φ::Function) where T
    @assert HIK.d == 1

    Dp, p = HIK.Dp, 1
    P = ContinuousPolynomial{0}(HIK.r)

    Rp = (P \ Dp)[Block.(1:p+1), Block.(1:p)]
    up = Rp*u

    λ = Rp * copy(λ) #(HIK.M \ λ)

    plan_P = plan_transform(P, (Block(p+1),), 1)
    oc = plan_P * max.(plan_P \ BlockVec(reshape(Vector(copy(up - pad(HIK.φ, axes(up,1)))),reverse(size(plan_P)))), 0)

    L = Legendre()
    x = axes(L,1)
    Dm = (Jacobi(2,2) \ (Derivative(x) * (Derivative(x) * L)))[1:p+1,1:p+1]
    Rm = (L \ Jacobi(2,2))[1:p+1,1:p+1]
    ΔP = Rm*Dm

    M = grammatrix(P)

    x, plan_P = plan_grid_transform(P, (Block(p+12),), 1)

    fP = plan_P * f.(x)
    φP = plan_P * φ.(x)

    MeshAdaptivity{T}(HIK, 1.0, up, zeros(1), λ, oc, fP, φP, Rp, ΔP, M)
end

function analyticity_coeffs(MA::MeshAdaptivity{T}) where T
    PG = MA.PG
    p, n = PG.p, PG.Nh
    # p = maximum(p)
    
    up = MA.up
    σs = T[]

    pv = p isa Integer ? repeat([p], n) : p
    for i in 1:n
        F = hcat(0.0:1:pv[i]+1, ones(pv[i]+2))
        m = F \ abs.(log.(abs.(up[i:n:end][1:pv[i]+2])))
        push!(σs, exp(-first(m)))
    end
    # σs[isnan.(σs)] .= 0.0
    return σs
end

function error_estimates(MA::MeshAdaptivity{T}; pg::Bool=false) where T
    PG = MA.PG
    n = PG.Nh

    if PG isa Union{<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}}
        p = PG.p
        r = PG.P.points
    elseif PG isa HIKSolver{T}
        p = 0
        r = PG.r
    end

    exp_ψP = MA.exp_ψP
    up,λ,oc = MA.up, MA.λ, MA.oc
    
    ΔP, M = MA.ΔP, MA.M

    ϵs = T[]

    if PG isa AdaptiveObstacleProblem
        ps = p[:]
        p = maximum(p)
        fc = pad_ψ(PG, PG.f)
        λ = pad_ψ(PG, λ)
        φc = pad_ψ(PG, PG.φ)
    else
        ps = p
        fc = PG.f
        φc = PG.φ
    end

    fP = MA.fP[Block.(1:p+12)]
    φP = MA.φP[Block.(1:p+12)]
    MP = M[Block.(1:p+12), Block.(1:p+12)]
    Mb = M[Block.(1:p+2), Block.(1:p+2)]

    for K in 1:n
        fv = pad(fc,axes(up,1))[K:n:end]
        fPv = fP[K:n:end]
        λv = pad(λ,axes(up,1))[K:n:end]
        uv = up[K:n:end]
        φv = pad(φc,axes(up,1))[K:n:end]
        φPv = φP[K:n:end]
        

        hT = r[K+1]-r[K]
        pT = ps isa Integer ? ps + 1 : ps[K] + 1
        
        dv = (fv + 4/hT^2*ΔP*uv + λv)
        MPv = MP[K:n:end, K:n:end]
        Mv = Mb[K:n:end, K:n:end]

        df = fPv - pad(fv, axes(fPv,1))
        dφ = φPv - pad(φv, axes(fPv,1))
        osc = hT^2/pT^2 *(df' * MPv * df)  + dφ' * MPv * dφ

        if pg
            @assert PG isa Union{<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}}
            exp_ψPv = exp_ψP[K:n:end]
            # ev = pad(uv, axes(exp_ψPv,1)) - pad(φv, axes(exp_ψPv,1)) + exp_ψPv
            # Res = sqrt(MA.α^2 * hT^2/pT^2 * dv' * Mv  * dv + ev' * MPv * ev + osc)

            ev = uv - φv + exp_ψPv
            Res = sqrt(MA.α^2 * hT^2/pT^2 * dv' * Mv  * dv + ev' * Mv * ev + osc)
        else
            ov = oc[K:n:end]
            Res = sqrt(hT^2/pT^2 * dv' * Mv  * dv + ov' * Mv * ov + abs(λv' * Mv * (φv-uv)) + osc)
        end
        push!(ϵs, Res)
    end
    ϵs

end

function h_refine(MA::MeshAdaptivity{T}, ϵs::AbstractVector{T}; δ::T=0.1) where T
    refine_idx = findall(ϵs .> δ*maximum(ϵs))
    PG = MA.PG
    if PG isa Union{<:ObstacleProblem{T}, <:AdaptiveObstacleProblem{T}}
        r = PG.P.points
    elseif PG isa HIKSolver{T}
        r = PG.r
    end
    sort(r ∪ (r[refine_idx.+1]+r[refine_idx])/2)
end

function hp_refine(MA::MeshAdaptivity{T}, ϵs::AbstractVector{T}, σs::AbstractVector{T}; σ::T=0.5, δ::T=0.1, dp::Integer=2) where T
    refine_idx = findall(ϵs .> δ*maximum(ϵs))

    r = MA.PG.P.points
    p = copy(MA.PG.p)
    r_ = copy(r)
    for idx in refine_idx
        # idxs = idx-1:idx+1
        if σs[idx] < σ
            p[idx] = p[idx].+dp
        else
            r_ = sort(r_ ∪ (r[idx.+1]+r[idx]) ./2 )
            # min(p[idx],p[idx+1]), 
            min_p = minimum(p)
            p = vcat(p[1:idx],min_p,p[idx+1:end])
        end
    end
    p[findall(p .< maximum(p)-3)] .=  maximum(p)-3
    (r_,p)
end

function uniform_refine(r::AbstractVector{T}) where T
    sort(r ∪ (r[1:end-1]+r[2:end])/2)
end

function hp_uniform_refine(r::AbstractVector{T}, p::AbstractVector{Int64}, σs::AbstractVector{T};σ::T=0.5,dp::Integer=2) where T
    pidx = findall(σs .< σ)
    p[pidx] .+= dp

    ridx = findall(σs .≥ σ)
    r_ = sort(r ∪ (r[ridx.+1]+r[ridx])/2)
    p_ = zeros(Int64, lastindex(r_)-1)

    ridx_ = ridx + Vector(0:lastindex(ridx)-1)
    p_[ridx_] = p[ridx]
    p_[ridx_ .+ 1] .= p[ridx]
    p_[p_ .== 0] = p[pidx]

    p_[findall(p_ .< maximum(p_)-3)] .=  maximum(p_)-3
    (r_, p_)
end
