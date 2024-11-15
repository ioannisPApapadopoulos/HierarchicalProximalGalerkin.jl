using MKL
using HierarchicalProximalGalerkin
using SpecialFunctions
using Plots, DelimitedFiles

f(x,y) = 100.0
φ(x,y) = (besselj(0,20x)+1)*(besselj(0,20y)+1)

path = "output/bessel_obstacle/"
if !isdir(path)
    mkpath(path)
end

function save_data(ndofs, tics, avg_tics, h1s, subpath)
    writedlm(path*subpath*"_ndofs.log", ndofs)
    writedlm(path*subpath*"_avg_tics.log", avg_tics)
    writedlm(path*subpath*"_tics.log", tics)
    writedlm(path*subpath*"_h1s.log", h1s)
end

T = Float64
r = range(0,1,11)
us, rs = Vector{T}[], AbstractVector{T}[]
newton_its = Int64[]
tics = T[]
ndofs = Int64[]
HIKs = []
for iters = 1:8
    print("Mesh level: $iters.\n")
    HIK = HIKSolver2D(r, f, φ)
    push!(HIKs, HIK)
    push!(ndofs, lastindex(HIK.A,1))
    if iters == 1
        u0 = zeros(lastindex(HIK.A,1))
    else
        xy, plan_D = plan_grid_transform(HIK.Dp, Block(1,1))
        ud(x,y) = evaluate2D(us[iters-1], x, y, 1, HIKs[iters-1].Dp)
        x,y=first(xy),last(xy)
        u0 = (plan_D * ud.(x,reshape(y,1,1,size(y)...)))[:]
    end

    tic = @elapsed u, λ, its = HierarchicalProximalGalerkin.solve(HIK, u0, show_trace=false, tol=1e-7);
    writedlm(path*"bessel_u_$iters.log", u)
    push!(us, u); push!(newton_its, its); push!(tics, tic)
    r = uniform_refine(r)
    push!(rs, r)
end
avg_tics = tics ./ newton_its
# writedlm(path*"bessel_ndofs_hik.log", ndofs)
# writedlm(path*"bessel_avg_tics_hik.log", avg_tics)

u_ref = us[end]
Dp, A, M = HIKs[end].Dp, HIKs[end].A, HIKs[end].M
xy, plan_D = plan_grid_transform(Dp, Block(1,1))
x,y=first(xy),last(xy)
l2s, h1s = T[], T[]
for iters = 1:5
    print("\nComputing error, mesh level: $iters.\n")
    ud(x,y) = evaluate2D(us[iters], x, y, 1, HIKs[iters].Dp)
    d = u_ref - (plan_D * ud.(x,reshape(y,1,1,size(y)...)))[:]
    push!(l2s, sqrt(d' * (M * d)))
    push!(h1s, sqrt(d' * (A * d) + l2s[end]^2))
    # writedlm(path*"bessel_h1s_hik.log", h1s)
end
save_data(ndofs, tics, avg_tics, h1s, "hik_uniform")