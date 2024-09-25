function ls_fϕ(PG::Union{<:ObstacleProblem{T},<:AdaptiveObstacleProblem{T}, <:ObstacleProblem2D{T}, <:BCsObstacleProblem2D{T}}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}, ls_α::T) where T
    res_u, res_ψ = matrixfree_residual(PG, u+ls_α*du, ψ+ls_α*dψ, w, α)
    LinearAlgebra.norm([res_u;res_ψ])
end

function ls_fdϕ(PG::ObstacleProblem{T}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}, ls_α::T) where T
    res_u, res_ψ = matrixfree_residual(PG, u+ls_α*du, ψ+ls_α*dψ, w, α)
    A, B = PG.A, PG.B
    dot([res_u;res_ψ], [α*A*du + B*dψ;-B'*du + apply_D_alias(PG,dψ,ψ)]) / LinearAlgebra.norm([res_u;res_ψ])
end

function ls_fdϕ(PG::Union{<:ObstacleProblem2D{T},<:BCsObstacleProblem2D{T}}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}, ls_α::T) where T
    res_u, res_ψ = matrixfree_residual(PG, u+ls_α*du, ψ+ls_α*dψ, w, α)
    A, B = PG.A, PG.B
    dot([res_u;res_ψ], [α*A*du + B*dψ;-B'*du + apply_D(PG,dψ,ψ)]) / LinearAlgebra.norm([res_u;res_ψ])
end

function ls_fdϕ(PG::AdaptiveObstacleProblem{T}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}, ls_α::T) where T
    res_u, res_ψ = matrixfree_residual(PG, u+ls_α*du, ψ+ls_α*dψ, w, α)
    A, B = PG.A, PG.B
    dot([res_u;res_ψ], [α*A*du + B*dψ;-B'*du + apply_D_adaptive(PG,dψ,ψ)]) / LinearAlgebra.norm([res_u;res_ψ])
end

function pg_linesearch(PG::Union{<:ObstacleProblem{T},<:AdaptiveObstacleProblem{T}, <:ObstacleProblem2D{T}, <:BCsObstacleProblem2D{T}}, ls::BackTracking{T, Int64}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}) where T
    ls_ϕ = ls_α -> ls_fϕ(PG, u, ψ, w, α, du, dψ, ls_α)
    ls_ϕ0 = ls_ϕ(0.0)
    ls_dϕ0 = ls_fdϕ(PG, u, ψ, w, α, du, dψ, 0.0)

    ls_α, _ =  ls(ls_ϕ, 1.0, ls_ϕ0, ls_dϕ0)
    u += ls_α * du
    ψ += ls_α * dψ
    u, ψ, ls_α
end



# function ls_fdϕ(PG::AdaptiveObstacleProblem{T}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}, ls_α::T) where T
#     res_u, res_ψ = matrixfree_residual(PG, u+ls_α*du, ψ+ls_α*dψ, w, α)
#     A, B = PG.A, PG.B
#     2*dot([res_u;res_ψ], [α*A*du + B*dψ;-B'*du + apply_D_adaptive(PG,dψ,ψ)])
# end

# function pg_linesearch(PG::AdaptiveObstacleProblem{T}, ls::BackTracking{T, Int64}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}) where T
#     ls_ϕ = ls_α -> ls_fϕ(PG, u, ψ, w, α, du, dψ, ls_α)
#     ls_ϕ0 = ls_ϕ(0.0)
#     ls_dϕ0 = ls_fdϕ(PG, u, ψ, w, α, du, dψ, 0.0)

#     ls_α, _ =  ls(ls_ϕ, 1.0, ls_ϕ0, ls_dϕ0)
#     u += ls_α * du
#     ψ += ls_α * dψ
#     u, ψ, ls_α
# end



# function ls_fdϕ(PG::ObstacleProblem2D{T}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}, ls_α::T) where T
#     res_u, res_ψ = matrixfree_residual(PG, u+ls_α*du, ψ+ls_α*dψ, w, α)
#     A, B = PG.A, PG.B
#     2*dot([res_u;res_ψ], [α*A*du + B*dψ;-B'*du + apply_D(PG,dψ,ψ)])
# end

# function pg_linesearch(PG::ObstacleProblem2D{T}, ls::BackTracking{T, Int64}, u::AbstractVector{T}, ψ::AbstractVector{T}, w::AbstractVector{T}, α::T, du::AbstractVector{T}, dψ::AbstractVector{T}) where T
#     ls_ϕ = ls_α -> ls_fϕ(PG, u, ψ, w, α, du, dψ, ls_α)
#     ls_ϕ0 = ls_ϕ(0.0)
#     ls_dϕ0 = ls_fdϕ(PG, u, ψ, w, α, du, dψ, 0.0)

#     ls_α, _ =  ls(ls_ϕ, 1.0, ls_ϕ0, ls_dϕ0)
#     u += ls_α * du
#     ψ += ls_α * dψ
#     u, ψ, ls_α
# end