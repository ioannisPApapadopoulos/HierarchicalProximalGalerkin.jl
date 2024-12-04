using MKL
using HierarchicalProximalGalerkin
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings, DelimitedFiles
using SpecialFunctions

"""
This example requires the package ```SpecialFunctions.jl```

Section 7.2: Solve a 2D obstacle problem with a bessel-type oscillatory obstacle.

Here we use the hpG solver with sparse LU factorization. We consider
    (i)   p-uniform refinement
    (ii)  h-uniform refinement with p=2, 3
    (iii) hp-uniform refinement

"""
T = Float64
f(x,y) = 100.0
φ(x,y) = (besselj(0,20x)+1)*(besselj(0,20y)+1)

path = "output/bessel_obstacle/"
if !isdir(path)
    mkpath(path)
end
function save_data(ndofs, tics, avg_tics, h1s, its, subpath)
    writedlm(path*subpath*"_ndofs.log", ndofs)
    writedlm(path*subpath*"_avg_tics.log", avg_tics)
    writedlm(path*subpath*"_tics.log", tics)
    writedlm(path*subpath*"_h1s.log", h1s)
    writedlm(path*subpath*"_its.log", its)
end

function bessel_solve(r::AbstractVector{T}, p::Int, f::Function, φ::Function; its_max::Int=6, show_trace::Bool=false) where T
    PG = ObstacleProblem2D(r, p, f, φ, matrixfree=false);
    αs = [Vector(2.0.^(-7:0.5:-3)); 2^(-3)]

    # Md = sparse(grammatrix(PG.Dp)[Block.(oneto(p+1)), Block.(oneto(p+1))])
    # Md = kron(Md, Md)

    tic = @elapsed u, ψ, iters = pg_hierarchical_solve(PG, αs,
        matrixfree=false,backtracking=false,its_max=its_max,show_trace=show_trace,
        c_1=-1e4,pf_its_max=2,β=0.0)
    return PG, (u, ψ), tic, iters
end

"""
p-uniform refinement, fixed h
"""
r = range(0,1,11)

newton_its_p_fem, tics_p_fem = Int32[], T[]
ndofs_p_fem, us, PGs_p_fem = Int64[], Vector{T}[], []
# Solve with p-refinement
for p in 1:25
    print("Considering p=$p.\n")

    PG, z, tic, iters = bessel_solve(r, p, f, φ,show_trace=false)

    push!(PGs_p_fem, PG);
    push!(ndofs_p_fem, size(PG.A, 1))
    push!(us, z[1])
    push!(tics_p_fem, tic)
    push!(newton_its_p_fem, iters[1])
end
avg_tics_p_fem = tics_p_fem ./ newton_its_p_fem

# Approximate errors
u_ref = reshape(us[end], isqrt(lastindex(us[end])), isqrt(lastindex(us[end])))
A = PGs_p_fem[end].A
KR = Block.(oneto(PGs_p_fem[end].p+1))
M1 = sparse(Symmetric((PGs_p_fem[end].Dp' * PGs_p_fem[end].Dp)[KR,KR]))
M = Symmetric(kron(M1,M1))
l2s_p_fem, h1s_p_fem = T[], T[]
for iters = 1:length(us)-1
    print("\nComputing error, mesh level: $iters.\n")
    d = u_ref - zero_pad(reshape(us[iters], isqrt(lastindex(us[iters])), isqrt(lastindex(us[iters]))), isqrt(lastindex(u_ref)))
    d=d[:]
    push!(l2s_p_fem, sqrt(d' * (M * d)))
    push!(h1s_p_fem, sqrt(d' * (A * d) + l2s_p_fem[end]^2))
end
save_data(ndofs_p_fem, tics_p_fem, avg_tics_p_fem, h1s_p_fem, newton_its_p_fem, "p_uniform")

"""
h-uniform refinement, fixed p
"""
for p in [1,2]
    us = Vector{T}[]
    PGs_h_fem,tics_h_fem = [], T[]
    newton_its_h_fem = Int32[]
    ndofs_h_fem = Int64[]
    Nmax = p==1 ? 7 : 6
    # Solve
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

    # Measure errors
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
        vals = evaluate2D(us[iters], x, y, PGs_h_fem[iters].p+1, PGs_h_fem[iters].Dp)
        d = (u_ref - plan_D * vals)[:]
        push!(l2s_h_fem, sqrt(d' * (M * d)))
        push!(h1s_h_fem, sqrt(d' * (A * d) + l2s_h_fem[end]^2))
    end
    save_data(ndofs_h_fem, tics_h_fem, avg_tics_h_fem, h1s_h_fem, newton_its_h_fem, "h_uniform_p_$(p+1)")
end

"""
hp-uniform refinement
"""

us = Vector{T}[]
PGs,tics = [], T[]
newton_its = Int32[]
ndofs = Int64[]
# Solve
for (N, p) in zip(1:6, 1:6)

    print("p=$p, mesh level: $N.\n")

    r = range(0,1,10*2^(N-1)+1)
    PG, z, tic, iters = bessel_solve(r, p, f, φ, show_trace=false)

    push!(us, z[1])
    push!(tics, tic)
    push!(newton_its, iters[1])
    push!(PGs, PG);
    push!(ndofs, size(PG.A, 1))
end
avg_tics = tics ./ newton_its

# Measure errors
u_ref = reshape(us[end], isqrt(lastindex(us[end])), isqrt(lastindex(us[end])))
A = PGs[end].A
KR = Block.(oneto(PGs[end].p+1))
M1 = sparse(Symmetric((PGs[end].Dp' * PGs[end].Dp)[KR,KR]))
M = Symmetric(kron(M1,M1))
xy, plan_D = plan_grid_transform(PGs[end].Dp, Block(PGs[end].p+1,PGs[end].p+1));
x,y=first(xy),last(xy);
l2s, h1s = T[], T[]
for iters = 1:lastindex(ndofs)-1
    print("Computing error, iter=$iters.\n")
    vals = evaluate2D(us[iters], x, y, PGs[iters].p+1, PGs[iters].Dp)
    d = (u_ref - plan_D * vals)[:]
    push!(l2s, sqrt(d' * (M * d)))
    push!(h1s, sqrt(d' * (A * d) + l2s[end]^2))
    # writedlm(path*"bessel_h1s_h_fem_p_$(p+1).log", h1s_h_fem)
end
save_data(ndofs, tics, avg_tics, h1s, newton_its, "hp_uniform")

if false
    """
    Plot solutions

    """
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
end