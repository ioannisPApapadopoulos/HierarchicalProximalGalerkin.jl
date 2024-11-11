using MKL
using HierarchicalProximalGalerkin
using Plots, LaTeXStrings, DelimitedFiles

include("high_freq_setup.jl")

T = Float64
r = range(0,1,11)

function high_freq_hik_solve(r::AbstractVector{T}, refinement_strategy::Function, nlevels::Int=12) where T
    us, rs = Vector{T}[], AbstractVector{T}[]

    newton_its = Int64[]
    tics = T[]
    ndofs = Int64[]
    HIKs = []
    for iters = 1:nlevels
        push!(rs, r)
        print("Mesh level: $iters.\n")
        HIK = HIKSolver(r, f, φ)
        push!(HIKs, HIK)
        push!(ndofs, lastindex(HIK.A,1))

        u0 = zeros(lastindex(HIK.A,1))

        tic = @elapsed u, λ, its = HierarchicalProximalGalerkin.solve(HIK, u0, show_trace=false, tol=1e-7);

        push!(us, u); 
        push!(newton_its, its); push!(tics, tic)
        if iters < nlevels
            MA = MeshAdaptivity(HIK, u, λ, f, φ);
            r = refinement_strategy(r, MA)
        end

    end
    avg_tics = tics ./ newton_its

    l2s_u, h1s_u = T[], T[]
    for iters = 1:nlevels

        Dp, bb = HIKs[iters].Dp, 3
        xg, plan_D = plan_grid_transform(Dp, (Block(bb),), 1);
        uD = pad(plan_D * ua.(xg), axes(Dp,2))
        Md = sparse(grammatrix(Dp)[Block.(1:bb), Block.(1:bb)])
        Ad = sparse(-weaklaplacian(Dp)[Block.(1:bb), Block.(1:bb)]);
        print("Computing error, mesh level: $iters.\n")
        d, l2d = l2_norm_u_fast(uD[1:size(Md,1)], pad(us[iters],1:size(Md,1)), Md)
        push!(l2s_u, l2d)
        push!(h1s_u, h1_norm_u_fast(d, Ad, l2s_u[end]))
    end
    return rs, ndofs, avg_tics, h1s_u
end

function hik_h_uniform_refine(r::AbstractVector, MA)
    uniform_refine(r)
end
rs, ndofs, avg_tics, h1s_u = high_freq_hik_solve(r,hik_h_uniform_refine,12)
save_data(ndofs, avg_tics, h1s_u, "hik")

function hik_h_adaptive_refine(r::AbstractVector, MA)
    ϵs = error_estimates(MA)
    h_refine(MA,ϵs,δ=0.5)
end
rs, ndofs, avg_tics, h1s_u = high_freq_hik_solve(r,hik_h_adaptive_refine,15) # 12
save_data(ndofs, avg_tics, h1s_u, "hik_adaptive")