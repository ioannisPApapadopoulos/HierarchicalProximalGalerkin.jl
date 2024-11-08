using MKL
using HierarchicalProximalGalerkin
using Plots, LaTeXStrings, DelimitedFiles

include("high_freq_setup.jl")

T = Float64
r = range(0,1,11)
us, rs = Vector{T}[], AbstractVector{T}[]
newton_its = Int64[]
tics = T[]
ndofs = Int64[]
HIKs = []
for iters = 1:12
    print("\nMesh level: $iters.\n")
    HIK = HIKSolver(r, f, φ)
    push!(HIKs, HIK)
    push!(ndofs, lastindex(HIK.A,1))
    # if iters == 1
    u0 = zeros(lastindex(HIK.A,1))
    # else
    #     xg, plan_D = plan_grid_transform(HIK.Dp, Block(1))
    #     ud(x) = (HIKs[iters-1].Dp*pad(us[iters-1], axes(HIKs[iters-1].Dp,2)))[x]
    #     # ud(x) = evaluate_u(HIKs[iters-1].Dp, x, us[iters-1])
    #     u0 = vec(plan_D * ud.(xg))
    # end

    tic = @elapsed u, λ, its = HierarchicalProximalGalerkin.solve(HIK, u0, show_trace=false, tol=1e-7);

    push!(us, u); push!(newton_its, its); push!(tics, tic)
    r = uniform_refine(r)
    push!(rs, r)
end
avg_tics = tics ./ newton_its

writedlm(path*"hik_ndofs.log", ndofs)
writedlm(path*"hik_avg_tics.log", avg_tics)

l2s_u, h1s_u = T[], T[]
for iters = 1:length(us)

    Dp, bb = HIKs[iters].Dp, 3
    xg, plan_D = plan_grid_transform(Dp, (Block(bb),), 1);
    uD = pad(plan_D * ua.(xg), axes(Dp,2))
    Md = sparse(grammatrix(Dp)[Block.(1:bb), Block.(1:bb)])
    Ad = sparse(-weaklaplacian(Dp)[Block.(1:bb), Block.(1:bb)]);
    print("\nComputing error, mesh level: $iters.\n")
    d, l2d = l2_norm_u_fast(uD[1:size(Md,1)], pad(us[iters],1:size(Md,1)), Md)
    push!(l2s_u, l2d)
    push!(h1s_u, h1_norm_u_fast(d, Ad, l2s_u[end]))


    # writedlm("bessel_h1s_hik.log", h1s)
end
writedlm(path*"hik_h1s.log", h1s_u)
Plots.plot(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10, label="(HIK) "*L"h"*"-uniform, "*L"p=1")