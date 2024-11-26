using MKL
using HierarchicalProximalGalerkin
using SparseArrays, LinearAlgebra
using Plots, LaTeXStrings
using BlockArrays


f(x,y) = 20.0
function φc(x,y,c)
    if abs(x-0.5) ≥ 0.25|| abs(y-0.5) ≥ 0.25
    # if abs(x-0.5) ≥ 0.25 && abs(y-0.5) ≤ 0.25 || abs(y-0.5) ≥ 0.25 && abs(x-0.5) ≤ 0.25
        return c
    else
        return 100.0
    end
end

T = Float64

path = "output/gradient-bound/"
if !isdir(path)
    mkpath(path)
end
function save_data(ndofs, tics, avg_tics, h1s, subpath)
    writedlm(path*subpath*"_ndofs.log", ndofs)
    writedlm(path*subpath*"_avg_tics.log", avg_tics)
    writedlm(path*subpath*"_tics.log", tics)
    writedlm(path*subpath*"_h1s.log", h1s)
end


function gradient_bound_solve(r::AbstractVector{T}, p::Int, c::T) where T
    φ(x,y) = φc(x,y,c)

    print("Mesh refinement level: $its.\n")
    PG = GradientBounds2D(r, p, f, φ) 
    # αs = [1e-1, 1e0, 1e1, 1e1,1e2,1e2]
    αs = [Vector(2.0.^(-7:0.5:2)); 2^(2)]
    u, ψ, w, iters = pg_hierarchical_solve(PG, αs,
        matrixfree=false,backtracking=false,its_max=4,
        pf_its_max=4,return_w=true,show_trace=true, β=0.0)
    
    return u
end

r = range(0,1,9)
u = gradient_bound_solve(r, 15, 0.5)
writedlm(path*"u_ref_p_uniform.log", u)

r = range(0,1,2^7+1)
u = gradient_bound_solve(r, 7, 0.5)
writedlm(path*"u_ref_h_uniform.log", u)
