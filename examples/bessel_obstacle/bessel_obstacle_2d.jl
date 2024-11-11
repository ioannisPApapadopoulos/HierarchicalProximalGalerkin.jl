using MKL
using HierarchicalProximalGalerkin
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings, DelimitedFiles
using SpecialFunctions

"""
This example requires the package ```SpecialFunctions.jl```
"""
f(x,y) = 100.0
φ(x,y) = (besselj(0,20x)+1)*(besselj(0,20y)+1)

path = "output/bessel_obstacle/"
if !isdir(path)
    mkpath(path)
end

T = Float64

function bessel_solve(r::AbstractVector{T}, p::Int, f::Function, φ::Function; its_max::Int=6, show_trace::Bool=false) where T
    PG = ObstacleProblem2D(r, p, f, φ);
    
    # αs = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e3, 1e3]
    αs = 2.0.^(-7:6) # 2.0.^(-7:10)

    # Md = sparse(grammatrix(PG.Dp)[Block.(oneto(p+1)), Block.(oneto(p+1))])
    # Md = kron(Md, Md)

    tic = @elapsed u, ψ, iters = pg_hierarchical_solve(PG, αs,
        matrixfree=false,backtracking=false,its_max=its_max,show_trace=show_trace,
        c_1=-1e4,pf_its_max=2,β=0.0)
    return PG, (u, ψ), tic, iters
end

### p-refinement, fixed h
newton_its_p_fem, tics_p_fem = Int32[], T[]
ndofs_p_fem = Int64[]
us = Vector{T}[]
PGs_p_fem = []
r = range(0,1,11)
pmax = 12
for p in 1:pmax
    print("Considering p=$p.\n")

    PG, z, tic, iters = bessel_solve(r, p, f, φ,show_trace=false)

    push!(PGs_p_fem, PG);
    push!(ndofs_p_fem, size(PG.A, 1))
    push!(us, z[1])
    push!(tics_p_fem, tic)
    push!(newton_its_p_fem, iters[1])
end
avg_tics_p_fem = tics_p_fem ./ newton_its_p_fem
writedlm(path*"bessel_avg_tics_p_fem.log", avg_tics_p_fem)
writedlm(path*"bessel_ndofs_p_fem.log", ndofs_p_fem)
writedlm(path*"bessel_newton_p_fem.log", newton_its_p_fem)

u_ref = reshape(us[end], isqrt(lastindex(us[end])), isqrt(lastindex(us[end])))
A = PGs_p_fem[end].A
KR = Block.(oneto(PGs_p_fem[end].p+1))
M1 = sparse(Symmetric((PGs_p_fem[end].Dp' * PGs_p_fem[end].Dp)[KR,KR]))
M = Symmetric(kron(M1,M1))
l2s_p_fem, h1s_p_fem = T[], T[]
for iters = 1:pmax-1
    print("\nComputing error, mesh level: $iters.\n")
    d = u_ref - zero_pad(reshape(us[iters], isqrt(lastindex(us[iters])), isqrt(lastindex(us[iters]))), isqrt(lastindex(u_ref)))
    d=d[:]
    push!(l2s_p_fem, sqrt(d' * (M * d)))
    push!(h1s_p_fem, sqrt(d' * (A * d) + l2s_p_fem[end]^2))
end
writedlm(path*"bessel_h1s_p_fem.log", h1s_p_fem)

### h-refinement, fixed p

for p in [1,2]
    us = Vector{T}[]
    PGs_h_fem,tics_h_fem = [], T[]
    newton_its_h_fem = Int32[]
    ndofs_h_fem = Int64[]
    Nmax = p==1 ? 5 : 4
    for N in 1:Nmax
        r = range(0,1,10*2^(N-1)+1)
        print("p=$p, mesh level: $N.\n")

        PG, z, tic, iters = bessel_solve(r, p, f, φ, show_trace=false)

        push!(us, z[1])
        push!(tics_h_fem, tic)
        push!(newton_its_h_fem, iters[1])
        push!(PGs_h_fem, PG);
        push!(ndofs_h_fem, size(PG.A, 1))
    end
    avg_tics_h_fem = tics_h_fem ./ newton_its_h_fem
    writedlm(path*"bessel_avg_tics_h_fem_p_$(p+1).log", avg_tics_h_fem)
    writedlm(path*"bessel_ndofs_h_fem_p_$(p+1).log", ndofs_h_fem)
    writedlm(path*"bessel_newton_h_fem_p_$(p+1).log", newton_its_h_fem)

    u_ref = reshape(us[end], isqrt(lastindex(us[end])), isqrt(lastindex(us[end])))
    A = PGs_h_fem[end].A
    KR = Block.(oneto(PGs_h_fem[end].p+1))
    M1 = sparse(Symmetric((PGs_h_fem[end].Dp' * PGs_h_fem[end].Dp)[KR,KR]))
    M = Symmetric(kron(M1,M1))
    xy, plan_D = plan_grid_transform(PGs_h_fem[end].Dp, Block(PGs_h_fem[end].p+1,PGs_h_fem[end].p+1))
    x,y=first(xy),last(xy)
    l2s_h_fem, h1s_h_fem = T[], T[]
    for iters = 1:Nmax-1
        print("Computing error, p=$p, mesh level: $iters.\n")
        ud(x,y) = evaluate2D(us[iters], x, y, PGs_h_fem[iters].p+1, PGs_h_fem[iters].Dp)
        d = (u_ref - plan_D * ud.(x,reshape(y,1,1,size(y)...)))[:]
        push!(l2s_h_fem, sqrt(d' * (M * d)))
        push!(h1s_h_fem, sqrt(d' * (A * d) + l2s_h_fem[end]^2))
        writedlm(path*"bessel_h1s_h_fem_p_$(p+1).log", h1s_h_fem)
    end
end

# Plot solution
xx = range(0,1,201)
Ux = evaluate2D(us[end], xx, xx, PGs_h_fem[end].p+1, PGs_h_fem[end].Dp)
Plots.gr_cbar_offsets[] = (-0.05,-0.01)
Plots.gr_cbar_width[] = 0.03
surface(xx,xx,Ux,
    color=:diverging,
    xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y)",
    margin=(-6, :mm),
)
Plots.savefig(path*"oscillatory_obstacle.pdf")

xx = range(0,1,200)
ux = evaluate2D(us[end], xx, [0.5], PGs[end].p+1, PGs[end].Dp)'
ox = φ.(xx, 0.5)
Plots.plot(xx, [ux ox],
    linewidth=2,
    label=[L"$u$" L"$\varphi$"],
    linestyle=[:solid :dash],
    xlabel=L"x",
    xlabelfontsize=20,
    xtickfontsize=10,ytickfontsize=10,legendfontsize=15,
    ylim=[0,1.25],
    title=L"Slice at $y=1/2$")
Plots.savefig(path*"oscillatory_obstacle_slice.pdf")