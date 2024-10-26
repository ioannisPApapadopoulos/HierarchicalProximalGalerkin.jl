
"""
Apply (2,2)-block, i.e. given a coefficient vector 𝐮 of a function u
compute the action of
    ⟨v, exp(-ψ) u⟩ ∀ v.
"""

function _apply_D(u::AbstractVector{T}, ψ::AbstractVector{T}, M::AbstractMatrix{T}, P::ContinuousPolynomial, plan_P::PiecewiseOrthogonalPolynomials.ApplyPlan{T}) where T
    ψx = plan_P \ BlockVec(reshape(copy(ψ),reverse(size(plan_P))))
    ux = plan_P \ BlockVec(reshape(copy(u),reverse(size(plan_P))))
    vals = exp.(-ψx) .* ux
    M * (plan_P * vals)
end

function _apply_D_2d(u::AbstractVector{T}, ψ::AbstractVector{T}, M::AbstractMatrix{T}, plan_P::ApplyPlan{T}) where T
    nx, px, ny, py = size(plan_P)

    ψx = plan_P \ BlockMatrix(reshape(ψ, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    ux = plan_P \ BlockMatrix(reshape(u, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    vals = exp.(-ψx) .* ux
    M * (plan_P * vals)[:]
end

function apply_D(PG::ObstacleProblem{T}, u::AbstractVector{T}, ψ::AbstractVector{T}) where T
    M, P, plan_P = PG.M, PG.P, PG.plan_P
    Vector(_apply_D(u, ψ, M, P, plan_P))
end

function apply_D(PG::Union{<:ObstacleProblem2D{T},<:BCsObstacleProblem2D{T}}, u::AbstractVector{T}, ψ::AbstractVector{T}) where T
    M, plan_P = PG.M, PG.plan_P
    Vector(_apply_D_2d(u, ψ, M, plan_P))
end

function apply_D_adaptive(PG::AdaptiveObstacleProblem{T}, u::AbstractVector{T}, ψ::AbstractVector{T}) where T
    M, plan_Ps, p = PG.M, PG.plan_Ps, PG.p
    up = unique(p)

    ψs, us = cellwise_decomposition(PG, ψ), cellwise_decomposition(PG, u)

    for i in 1:lastindex(plan_Ps)
        ups = up[i]
        plan_P = plan_Ps[i]

        idx = findall(p .== ups)
        tψs, tus = ψs[idx], us[idx]

        tψx = plan_P \ BlockVec(reduce(hcat, tψs)')
        # tψx[tψx .< -10] .= -10.0
        tux = plan_P \ BlockVec(reduce(hcat, tus)')
        vals = exp.(-tψx) .* tux
        ψc = plan_P * vals
        rψc = reshape(ψc, reverse(size(plan_P)))
        for j in 1:size(rψc,1) ψs[idx[j]] = rψc[j,:] end
    end
    M * cellwise_interlace(PG, ψs)
end

function apply_D(PG::GradientBounds2D{T}, u::AbstractVector{T}, ψ::AbstractVector{T}) where T
    M, plan_P = PG.M, PG.plan_P
    nx, px, ny, py = size(plan_P)

    ψx = plan_P \ BlockMatrix(reshape(ψ, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    ux = plan_P \ BlockMatrix(reshape(u, nx*px, ny*py), repeat([px], nx), repeat([py], ny))

    vals = (PG.φx ./ (sqrt.(one(T) .+ ψx.^2)).^3) .* ux
    Vector(M * (plan_P * vals)[:])
end

## For evaluating the residual
function evaluate_D(PG::ObstacleProblem{T}, ψ::AbstractVector{T}) where T
    M, plan_P = PG.M, PG.plan_P
    ψx = plan_P \ BlockVec(reshape(copy(Vector(ψ)),reverse(size(plan_P))))
    M * Vector(plan_P * exp.(-ψx))
end

function evaluate_D(PG::AdaptiveObstacleProblem{T}, ψ::AbstractVector{T}) where T
    M, plan_Ps, p = PG.M, PG.plan_Ps, PG.p
    up = unique(p)

    ψs = cellwise_decomposition(PG, ψ)

    for i in 1:lastindex(plan_Ps)
        ups = up[i]
        plan_P = plan_Ps[i]
        p .== ups

        idx = findall(p .== ups)
        tψs = ψs[idx]

        tψx = plan_P \ BlockVec(reduce(hcat, tψs)')
        # tψx[tψx .< -10] .= -10.0
        vals = exp.(-tψx)
        ψc = plan_P * vals
        rψc = reshape(ψc, reverse(size(plan_P)))
        for j in 1:size(rψc,1) ψs[idx[j]] = rψc[j,:] end
    end
    M * cellwise_interlace(PG, ψs)
end

function evaluate_D(PG::Union{<:ObstacleProblem2D{T},<:BCsObstacleProblem2D{T}}, ψ::AbstractVector{T}) where T
    M, plan_P = PG.M, PG.plan_P
    nx, px, ny, py = size(plan_P)
    ψx = plan_P \ BlockMatrix(reshape(ψ, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    vals = exp.(-ψx)
    M * (plan_P * vals)[:]
end

function evaluate_D(PG::GradientBounds2D{T}, ψ::AbstractVector{T}) where T
    M, plan_P = PG.M, PG.plan_P
    nx, px, ny, py = size(plan_P)
    ψx = plan_P \ BlockMatrix(reshape(ψ, nx*px, ny*py), repeat([px], nx), repeat([py], ny))
    vals = (ψx .* PG.φx ./ (sqrt.(one(T) .+ ψx.^2)))
    M * (plan_P * vals)[:]
end

"""
Evaluate the residual at a coefficient vector [𝐮;ψ]
"""
function matrixfree_residual(PG::Union{<:ObstacleProblem{T},<:AdaptiveObstacleProblem{T},<:ObstacleProblem2D{T}}, 
                u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::Number) where T
    A, B, M = PG.A, PG.B, PG.M
    f, φ = PG.f, PG.φ

    x = α*A*u + B*(ψ - α*f - w)
    y = -B'*u - evaluate_D(PG, ψ) + M*φ
    (Vector(x), Vector(y))
end

function matrixfree_residual(PG::BCsObstacleProblem2D{T}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::Number) where T
    A, B, M = PG.A, PG.B, PG.M
    f, φ = PG.f, PG.φ
    bcs_Fx, bcs_Fy = PG.bcs_Fx, PG.bcs_Fy
    
    x = α*A*u + B*(ψ - α*f - w)
    y = -B'*u - evaluate_D(PG, ψ) + M*φ
    (Vector(x+α*bcs_Fx), Vector(y+bcs_Fy))
end

function matrixfree_residual(PG::GradientBounds2D, 
                u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::Number) where T
    A, B, G = PG.A, PG.B, PG.G
    f= PG.f

    x = α*A*u + B*(ψ - w) - α*G*f
    y = -B'*u + evaluate_D(PG, ψ)
    (Vector(x), Vector(y))
end