using MKL
using HierarchicalProximalGalerkin
using SparseArrays, LinearAlgebra
using Plots, LaTeXStrings
using BlockArrays
using DelimitedFiles

"""
Section 7.3: Solve a 2D obstacle problem with a bessel-type oscillatory obstacle.

Here we use the hpG solver with sparse LU factorization. We consider
    (i)   p-uniform refinement
    (ii)  h-uniform refinement with p=1,2,3
    (iii) hp-uniform refinement

"""

f(x,y) = 20.0
function φc(x,y,c)
    if abs(x-0.5) ≥ 0.25 || abs(y-0.5) ≥ 0.25
        return c
    else
        return 1e10
    end
end

T = Float64

path = "output/gradient-bound/"
if !isdir(path)
    mkpath(path)
end
function save_data(ndofs, tics, avg_tics, h1s, its, subpath)
    writedlm(path*subpath*"_ndofs.log", ndofs)
    writedlm(path*subpath*"_avg_tics.log", avg_tics)
    writedlm(path*subpath*"_tics.log", tics)
    writedlm(path*subpath*"_h1s.log", h1s)
    writedlm(path*subpath*"_newton_its.log", its)
end

function approximate_error(PG, u, Dp, p, u_ref)
    u_ref = reshape(u_ref, isqrt(lastindex(u_ref)), isqrt(lastindex(u_ref)))

    KR = Block.(oneto(p))
    M1 = sparse(Symmetric((Dp' * Dp)[KR,KR]))
    M = Symmetric(kron(M1,M1))
    Δ = weaklaplacian(Dp)
    A1 = sparse(Symmetric(-parent(Δ)[KR,KR]))
    A = sparse(kron(A1,M1) + kron(M1,A1))
    xy, plan_D = plan_grid_transform(Dp, Block(p,p))
    x,y=first(xy),last(xy)

    vals = evaluate2D(u, x, y, PG.p, PG.Dp)
    d = (u_ref - plan_D * vals)[:]
    l2 = sqrt(d' * (M * d))
    h1 = sqrt(d' * (A * d) + l2^2)

    return l2, h1
end

function gradient_bound_solve(r::AbstractVector{T}, p::Int, c::T, (u_ref, Dp_ref, p_ref, subpath); 
        nlevels::Int=1, 
        refinement_strategy::Function=x->0.0) where T

    φ(x,y) = φc(x,y,c)

    ndofs, tics = Int[], T[]
    rs, ps, iter_count, gmres_count = AbstractVector{T}[], typeof(p)[], Int[], Int[]
    us = []
    h1s = T[]

    for its in 1:nlevels
        push!(rs, r)
        push!(ps, p)
        print("Mesh refinement level: $its.\n")
        PG = GradientBounds2D(r, p, f, φ)

        # Md = sparse(grammatrix(PG.Dp)[Block.(oneto(p)), Block.(oneto(p))])
        # Md = kron(Md, Md)
        push!(ndofs, lastindex(PG.A,1))

        
        # αs = [1e-1, 1e0, 1e1, 1e1,1e2,1e2]
        αs = [Vector(2.0.^(-7:0.5:2)); 2^(2)]
        tic = @elapsed u, ψ, w, iters = pg_hierarchical_solve(PG, αs,
            matrixfree=false,backtracking=false,its_max=4,
            pf_its_max=4,return_w=true,show_trace=false, β=0.0)
        
        push!(us, u)
        push!(tics, tic)
        push!(iter_count, first(iters))
        push!(gmres_count, last(iters))

        l2, h1 = approximate_error(PG, u, Dp_ref, p_ref, u_ref)
        push!(h1s, h1)

        save_data(ndofs, tics, tics ./ iter_count, h1s, iter_count, subpath)
        if its < nlevels
            r, p = refinement_strategy(r, p)
        end
    end
    return us, ndofs, (iter_count, gmres_count), rs, ps, tics
end

φ(x,y) = φc(x,y,1/2)
function p_refine(r,p)
    r, p+1
end

"""
p-uniform refinement, fixed h
"""
try
    u_ref = readdlm(path*"reference/u_ref_p_uniform.log")[:,1]
catch e
    error("Need to compute reference solutions first.\n")
end
Dp = DirichletPolynomial(range(0,1,9))

r = range(0,1,9)
c = 1/2
us, ndofs, iters, rs, ps, tics = gradient_bound_solve(r, 1, c, nlevels=20, refinement_strategy=p_refine, 
                                                        (u_ref, Dp, 25, "p_uniform"));

"""
h-uniform refinement, fixed p
"""
u_ref = readdlm(path*"reference/u_ref_h_uniform.log")[:,1]
Dp = DirichletPolynomial(range(0,1,2^7+1))

function h_refine(r,p)
    range(0,1,2*r.len-1), p
end
c = 1/2
for (p, N) in zip([1,2,3], [7,6,6])
    r = range(0,1,5)
    us, ndofs, iters, rs, ps, tics = gradient_bound_solve(r, p, c, nlevels=N,
                    refinement_strategy=h_refine, (u_ref, Dp, 6, "_h_uniform_p_$p"));
end

"""
hp-uniform refinement
"""
function hp_refine(r,p)
    range(0,1,2*r.len-1), p+1
end
r = range(0,1,5)
us, ndofs, iters, rs, ps, tics = gradient_bound_solve(r, 1, c, nlevels=5,
        refinement_strategy=hp_refine, (u_ref, Dp, 6, "hp_uniform"));



"""
 Plot solutions
"""
if false
    Plots.gr_cbar_offsets[] = (-0.05,-0.01)
    Plots.gr_cbar_width[] = 0.03
    r= range(0,1,9)
    Dp = DirichletPolynomial(r)
    xx = range(0,1,201)

    PG = GradientBounds2D(r, 5, f, φ)
    αs = [Vector(2.0.^(-7:0.5:2)); 2^(2)]
    u, ψ, iters = pg_hierarchical_solve(PG, αs,
        matrixfree=false,backtracking=false,its_max=4,
        pf_its_max=4,show_trace=false, β=0.0)
    Ux = evaluate2D(u, xx, xx, 5, Dp)
    c=0.5; i=1;
    surface(xx,xx,Ux,
        color=:diverging,
        xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y)",
        # title=L"c="*"$(ct[i])",
        margin=(-6, :mm),
        xlabelfontsize=15, ylabelfontsize=15,zlabelfontsize=15,
        xticks=0:0.25:1, yticks=0:0.25:1
    )
    Plots.savefig("gradient-constraint.pdf")

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
end