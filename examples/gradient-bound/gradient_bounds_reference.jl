using MKL
using HierarchicalProximalGalerkin
using DelimitedFiles

f(x,y) = 20.0
function φc(x,y,c)
    if abs(x-0.5) ≥ 0.25|| abs(y-0.5) ≥ 0.25
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

function gradient_bound_solve(r::AbstractVector{T}, p::Int, c::T) where T
    φ(x,y) = φc(x,y,c)
    PG = GradientBounds2D(r, p, f, φ) 
    αs = [Vector(2.0.^(-7:0.5:2)); 2^(2)]
    u, ψ, iters = pg_hierarchical_solve(PG, αs,
        matrixfree=false,backtracking=false,its_max=4,
        pf_its_max=4,show_trace=true, β=0.0)
    return u
end

r = range(0,1,9)
u = gradient_bound_solve(r, 25, 0.5)
writedlm(path*"u_ref_p_uniform.log", u)

r = range(0,1,2^7+1)
u = gradient_bound_solve(r, 6, 0.5)
writedlm(path*"u_ref_h_uniform.log", u)
