using HierarchicalProximalGalerkin, SparseArrays
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings

k = 10π
c = 2 * k^2
f(x) = c* sin(k*x)
x0 = 0.0382764753038300
a0 = -22.6216651405434
x1 = 0.05
x2 = 0.85
x3 = 0.853422907268890
a3 = 6.74353407719214
b3 = -6.74353407719214

# k = 5π
# c = 2 * k^2
# f(x) = c* sin(k*x)
# x0 = 0.0765529506076600
# a0 = -11.3108325702717
# x1 = 0.1 
# x2 = 0.9
# x3 = 0.923447049392340
# a3 = 11.3108325702717
# b3 = -11.3108325702717
"""
FindRoot[{c/k * Cos[k*x] + a, c/k*Cos[k*y] + a, c/k^2*Sin[k*x] + a*x + b - 1, c/k^2*Sin[k*y] + a*y + b - 1}, {{x, 0.005}, {y, 0.005}, {a, 0}, {b, 0}}, WorkingPrecision -> 15]
FindRoot[{c/k * Cos[k*x] + a, c/k*Cos[k*y] + a, c/k^2*Sin[k*x] + a*x + b - 1, c/k^2*Sin[k*y] + a*y + b - 1}, {{x, 0.995}, {y, 0.995}, {a, 0}, {b, 0}}, WorkingPrecision -> 15]
FindRoot[{c/k * Cos[k*x] + a,  c/k^2*Sin[k*x] + a*x - 1}, {{x, 0.004}, {a, 0}}, WorkingPrecision -> 15]
FindRoot[{c/k * Cos[k*x] + a,  c/k^2*Sin[k*x]+a*x+b-1, a+b},{{x,0.988},{a,0}, {b,0}},WorkingPrecision->15]
"""


φ(x) = 1.0
function ua(x::T) where T
    if x < x0
        return c / k^2 * sin(k*x) + a0*x
    elseif x0 ≤ x ≤ x1
        return 1.0
    elseif x1 ≤ x ≤ x2
        return c / k^2 * sin(k*x) - 1
    elseif x2 < x < x3
        return 1.0
    else
        return c / k^2 * sin(k*x) + a3*x + b3
    end
end


function high_freq_solve(r::AbstractVector{T}, p::Union{<:Int, Vector{<:Int}}, refinement_strategy::Function; nlevels::Int=6) where T


    l2s_u, ndofs_u, h1s_u, tics = T[], Int[], T[], T[]
    rs, ps, iter_count, gmres_count = Vector{T}[], eltype(p)[], Int[], Int[]

    for its in 1:nlevels
        push!(rs, r)
        push!(ps, p)
        print("Mesh refinement level: $its.\n")
        PG = ObstacleProblem(r, p, f, φ);
        push!(ndofs_u,lastindex(PG.A,1))
        
        Dp, bb = PG.Dp, 100
        xg, plan_D = plan_grid_transform(Dp, (Block(bb),), 1);
        uD = pad(plan_D * ua.(xg), axes(Dp,2))
        Md = sparse(grammatrix(Dp)[Block.(1:bb), Block.(1:bb)])
        Ad = sparse(-weaklaplacian(Dp)[Block.(1:bb), Block.(1:bb)]);

        αs = vcat([1e-3*2^i for i = 0:14], repeat([20],7))
        tic = @elapsed u, ψ, w, iters = pg_hierarchical_solve(PG, αs;its_max=50, 
                show_trace=true, pf_its_max=3, #gmres_baseline_tol=1e-10,
                matrixfree=true,  backtracking=true, return_w=true,c_1=-1e3)
        # push!(us, u); push!(ψs, ψ); push!(ws, w); push!(λs, (w-ψ)/αs[end])
        
        push!(tics, tic)
        push!(iter_count, first(iters))
        push!(gmres_count, last(iters))

        d, l2d = l2_norm_u_fast(uD[1:size(Md,1)], pad(u,1:size(Md,1)), Md)
        push!(l2s_u, l2d)
        push!(h1s_u, h1_norm_u_fast(d, Ad, l2s_u[end]))

        MA = MeshAdaptivity(PG, αs[end], u, ψ, w, f, φ);
        # σs = analyticity_coeffs(MA)
        r, p = refinement_strategy(r, p, MA)
    end
    return (l2s_u, h1s_u), ndofs_u, (iter_count, gmres_count), rs, ps, tics
end


    # r, p = hp_uniform_refine(r,p,σs,σ=0.1)

    # r = uniform_refine(r)
    # p = repeat([1],length(r)-1)[:]
    
    # r,p = hp_refine(MA,ϵs,σs, δ=δ, σ=0.1)

function pg_h_uniform_refine(r::AbstractVector, p::Int, MA)
    uniform_refine(r), p
end
(l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
    high_freq_solve(range(0,1,11), 3, pg_h_uniform_refine, nlevels=10)
Plots.plot(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)

function pg_h_adaptive_refine(r::AbstractVector, p::Int, MA)
    ϵs = error_estimates(MA, pg=false)
    # σs = analyticity_coeffs(MA)
    h_refine(MA,ϵs,δ=0.01), p
end

(l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
    high_freq_solve(range(0,1,11), 3, pg_h_adaptive_refine, nlevels=10)
Plots.plot!(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)

function pg_h_adaptive_refine_pg(r::AbstractVector, p::Int, MA)
    ϵs = error_estimates(MA, pg=true)
    # σs = analyticity_coeffs(MA)
    h_refine(MA,ϵs,δ=0.01), p
end
(l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
    high_freq_solve(range(0,1,11), 3, pg_h_adaptive_refine_pg, nlevels=10)
Plots.plot!(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)

function pg_h_adaptive_p_uniform_refine(r::AbstractVector, p::Int)
    ϵs = error_estimates(MA, pg=false)
    h_refine(MA,ϵs,δ=0.01), p+4
end

function pg_p_uniform_refine(r::AbstractVector, p::Int, MA, ϵs, σs, δ)
    r, p+4
end

(l2s_u, h1s_u), ndofs, iters, rs, ps, tics = 
    high_freq_solve(range(0,1,11), 3, pg_h_adaptive_p_uniform_refine, nlevels=10)

# Plots.plot(ndofs_hp, h1s_u_hp, marker=:square, linewidth=2)
Plots.plot!(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)
Plots.plot!(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)


Plots.plot(ndofs_hp_refine_pg_false, h1s_u_hp_refine_pg_false, marker=:diamond, linewidth=2)
Plots.plot!(ndofs_hp_refine_pg_true, h1s_u_hp_refine_pg_true, yaxis=:log10,
    linewidth=2,
    marker=:dtriangle,
    # label="h-adaptive",
    xlabel="dofs",ylabel=L"$\Vert u - u_h \Vert_{L^2}$")
Plots.plot!(ndofs_hp_uniform_refine_pg_false, h1s_u_hp_uniform_refine_pg_false, marker=:xcross, linewidth=2, xaxis=:log10)
Plots.plot!(ndofs_uniform_refine_pg_false, h1s_u_uniform_refine_pg_false, marker=:xcross, linewidth=2, xaxis=:log10)
Plots.plot!(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10)


i=4; Plots.plot(rs[i], zeros(size(rs[i])), marker=:circle)

xx = 0:0.001:1
# PG = AdaptiveObstacleProblem(rs[end], ps[end], f, φ);
i=3; vals = evaluate_u(PGs[i],xx,us[i])
Plots.plot(xx,vals)
Plots.plot!(xx,ua.(xx))
Plots.plot(xx, abs.(vals-ua.(xx)))


## Plotting
vals = evaluate_u(PGs[end],xx,us[end])
Plots.plot(xx,[vals ones(lastindex(xx))], xlabel=L"x", 
    xlabelfontsize=15, xtickfontsize=12, ytickfontsize=12, 
    label=[L"u(x)" L"\varphi(x)"],
    legendfontsize=12,
    linewidth=2,
    linestyle=[:solid :dash],
    legend=:bottomright,
)
Plots.savefig("osc-data.pdf")