using MKL
using HierarchicalProximalGalerkin, SparseArrays
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings, DelimitedFiles

"""
Section 7.1: Solve a 1D obstacle problem with an oscillatory right-hand side

Here we use the hpG solver with sparse LU factorization. We consider
    (i)   h-adaptive, p-uniform
    (ii)  hp-adaptive
    (iii) h-uniform, p=2,4
    (iv)  hp-uniform refinement
    (v)   p-uniform, h=0.05

"""

include("1d_high_freq_setup.jl")

function high_freq_solve(r::AbstractVector{T}, p::Union{<:Int, Vector{<:Int}}, 
    refinement_strategy::Function; nlevels::Int=6, tol::T=1e-10, pf_max_its::Int=2) where T

    l2s_u, ndofs_u, h1s_u, tics = T[], Int[], T[], T[]
    rs, ps, iter_count, gmres_count = Vector{T}[], typeof(p)[], Int[], Int[]
    PG, us = [], []

    for its in 1:nlevels
        push!(rs, r)
        push!(ps, p)
        print("Mesh refinement level: $its.\n")
        PG = p isa Int ? ObstacleProblem(r, p, f, φ) : AdaptiveObstacleProblem(r, p, f, φ)
        push!(ndofs_u,lastindex(PG.A,1))
        
        Dp, bb = PG.Dp, 42
        xg, plan_D = plan_grid_transform(Dp, (Block(bb),), 1);
        uD = pad(plan_D * ua.(xg), axes(Dp,2))
        Md = sparse(grammatrix(Dp)[Block.(1:bb), Block.(1:bb)])
        Ad = sparse(-weaklaplacian(Dp)[Block.(1:bb), Block.(1:bb)]);


        αs = [Vector(2.0.^(-7:0.5:-3)); 2^(-3)]

        tic = @elapsed u, ψ, w, iters = pg_hierarchical_solve(PG, αs;its_max=6, 
                show_trace=false, pf_its_max=pf_max_its, #gmres_baseline_tol=1e-10,
                matrixfree=false, gmres_baseline_tol=1e-12, backtracking=true, return_w=true, c_1=-1e4,
                tolerance=tol, β=1e-8)

        push!(us, u)
        
        push!(tics, tic)
        push!(iter_count, first(iters))
        push!(gmres_count, last(iters))

        d, l2d = l2_norm_u_fast(uD[1:size(Md,1)], pad(u,1:size(Md,1)), Md)
        push!(l2s_u, l2d)
        push!(h1s_u, h1_norm_u_fast(d, Ad, l2s_u[end]))

        if its < nlevels
            MA = MeshAdaptivity(PG, αs[end], u, ψ, w, f, φ);
            r, p = refinement_strategy(r, p, MA)
        end
    end
    return us, (l2s_u, h1s_u), ndofs_u, (iter_count, gmres_count), rs, ps, tics
end

"""
h-adaptive, p-uniform
"""
function pg_h_adaptive_p_uniform_refine(r::AbstractVector, p::Int, MA)
    ϵs = error_estimates(MA, pg=false)
    h_refine(MA,ϵs,δ=0.3), p+1
end
p=3
us, (l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
    high_freq_solve(range(0,1,11), p, pg_h_adaptive_p_uniform_refine, nlevels=9)
save_data(ndofs, tics, tics ./ iters[1], h1s_u, "pg_h_adaptive_p_uniform_refine_p_$(p)")

"""
hp-adaptive
"""
function pg_hp_refine(r::AbstractVector, p::Vector{<:Int}, MA)
    ϵs = error_estimates(MA, pg=false)
    σs = analyticity_coeffs(MA)
    hp_refine(MA,ϵs,σs, δ=0.7, σ=0.8)
end
us, (l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
    high_freq_solve(range(0,1,11), [1 for _ in 1:10], pg_hp_refine, nlevels=10, pf_max_its=3)
save_data(ndofs, tics, tics ./ iters[1], h1s_u, "pg_hp_refine")
hs = rs[end][2:end]-rs[end][1:end-1]
minimum(hs), maximum(hs)
minimum(ps[end]), maximum(ps[end])


"""
h-uniform, p=2,4
"""
function pg_h_uniform_refine(r::AbstractVector, p::Int, MA)
    uniform_refine(r), p
end
for p in [1,3]
    nlevels =  p≈1 ? 8 : 7
    us, (l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
        high_freq_solve(range(0,1,11), p, pg_h_uniform_refine, nlevels=nlevels)
    save_data(ndofs, tics, tics ./ iters[1], h1s_u, "pg_h_uniform_refine_p_$p")
end

"""
p-uniform, h=0.05
"""
function pg_p_uniform_refine(r::AbstractVector, p::Int, MA)
    r, p+1
end
us, (l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
    high_freq_solve(range(0,1,21), 1, pg_p_uniform_refine, nlevels=40)
save_data(ndofs, tics, tics ./ iters[1], h1s_u, "pg_p_uniform_refine")

"""
pg error estimator alternative
"""
if false
    function pg_hp_refine(r::AbstractVector, p::Vector{<:Int}, MA)
        ϵs = error_estimates(MA, pg=true)
        σs = analyticity_coeffs(MA)
        hp_refine(MA,ϵs,σs, δ=0.3, σ=0.8)
    end
    us, (l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
        high_freq_solve(range(0,1,11), [1 for _ in 1:10], pg_hp_refine, nlevels=7)
    save_data(ndofs, tics ./ iters[1], h1s_u, "pg_hp_refine_pg_true")
    function pg_h_adaptive_refine_pg(r::AbstractVector, p::Int, MA)
        ϵs = error_estimates(MA, pg=true)
        h_refine(MA,ϵs,δ=0.7), p
    end
    p = 1
    (l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
        high_freq_solve(range(0,1,11), p, pg_h_adaptive_refine_pg, nlevels=10)
    save_data(ndofs, tics ./ iters[1], h1s_u, "pg_h_adaptive_refine_p_$(p)_pg_true")
end

"""
Solution plotting
"""
if false
    xx = range(0,1,201)
    Plots.plot(xx,[ua.(xx) ones(lastindex(xx))], xlabel=L"x", 
        xlabelfontsize=15, xtickfontsize=12, ytickfontsize=12, 
        label=[L"u(x)" L"\varphi(x)"],
        legendfontsize=12,
        linewidth=2,
        linestyle=[:solid :dash],
        legend=:bottomright,
    )
    Plots.savefig("osc-data.pdf")
end