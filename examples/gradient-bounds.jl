using HierarchicalProximalGalerkin
using SparseArrays, LinearAlgebra
using Plots, LaTeXStrings
using BlockArrays


f(x,y) = 20.0
function φc(x,y,c)
    if 0 ≤ x ≤ 0.25 || 0.75 ≤ x ≤ 1.0 || 0 ≤ y ≤ 0.25 || 0.75 ≤ y ≤ 1.0
        return c
    else
        return 100.0
    end
end

T = Float64

# gmres_its, newton_its, tics = Int32[], Int32[], T[]
# rs, ps = Vector{T}[], T[]
# ndofs = Int64[]
# l2s, h1s = T[], T[]
# PG, u, ψ, w = [], [], [], []

    # print("Considering mesh refinement $iter.\n")
function gradient_bound_solve(r::AbstractVector{T}, p::Int, c::T) where T
    φ(x,y) = φc(x,y,c)
    PG = GradientBounds2D(r, p, f, φ);
    Md = sparse(grammatrix(PG.Dp)[Block.(oneto(p)), Block.(oneto(p))])
    Md = kron(Md, Md)

    push!(ndofs, lastindex(PG.A,1))
    push!(rs, r)
    push!(ps, p)
    
    αs = [1e-1, 1e0, 1e1, 1e1,1e2,1e2]
    tic = @elapsed u, ψ, w, iters = pg_hierarchical_solve(PG, αs,
        matrixfree=false,backtracking=false,its_max=4,
        pf_its_max=4,return_w=true,show_trace=true, β=0.0, c_1=-1e4, Md=Md)
    return u, iters
end

Plots.gr_cbar_offsets[] = (-0.05,-0.01)
Plots.gr_cbar_width[] = 0.03
Dp = DirichletPolynomial(r)
xx = range(0,1,201)
r= range(0,1,9)
p=6

ct = ["1/2", "1", "2"]
us = []
for (c, i) in zip([0.5, 1.0, 2.0], 1:3)
    u, iters = gradient_bound_solve(r, p, c)
    push!(us, u)
    Ux = evaluate2D(u, xx, xx, p, Dp)

    surface(xx,xx,Ux,
        color=:diverging,
        xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y)",
        title=L"c="*"$(ct[i])",
        margin=(-6, :mm),
        xlabelfontsize=15, ylabelfontsize=15,zlabelfontsize=15,
    )
    Plots.savefig("gradient-constraint-$c.pdf")
end




y = 0.5
xx = range(0,1,501)
pt = Plots.plot()
for i in 1:3
    ux = evaluate2D(us[i],xx,[y], p, Dp)'
    pt = Plots.plot!(xx, [ux],
        linewidth=2,
        label=L"c="*"$(ct[i])",
        xlabel=L"x",
        ylabel=L"u(x,1/2)",
        title=L"Slice at $y=1/2$",
        xlabelfontsize=20,ylabelfontsize=18,
        legendfontsize=12,
    )
end
display(pt)
Plots.savefig("gradient-constraint-slice.pdf")