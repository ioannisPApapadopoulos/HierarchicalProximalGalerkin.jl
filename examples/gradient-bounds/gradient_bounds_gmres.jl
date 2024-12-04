using MKL
using HierarchicalProximalGalerkin
using SparseArrays, LinearAlgebra
using Plots, LaTeXStrings
using BlockArrays
using DelimitedFiles

"""
Section 7.3: Solve a generalized elastic-plastic torsion problem via the hpG solver.

In this script we use a preconditioned GMRES strategy with
    (i)   p-uniform refinement
    (ii)  h-uniform refinement with p=1,2,3
    (iii) hp-uniform refinement

"""

f(x,y) = 20.0
function φ(x,y)
    if abs(x-0.5) ≥ 0.25 || abs(y-0.5) ≥ 0.25
        return 0.5
    else
        return 1e10
    end
end

T = Float64

path = "output/gradient-bound/gmres/"
if !isdir(path)
    mkpath(path)
end
function save_data(tics, avg_tics, newton_its, avg_gmres_its, subpath)
    writedlm(path*subpath*"_avg_tics.log", avg_tics)
    writedlm(path*subpath*"_tics.log", tics)
    writedlm(path*subpath*"_newton_its.log", newton_its)
    writedlm(path*subpath*"_avg_gmres_its.log", avg_gmres_its)
end


function gradient_bound_solve(r::AbstractVector{T}, p::Int, f::Function, φ::Function; gmres_tol::T=1e-5, β::T=1e-6,  show_trace::Bool=false) where T

    PG = GradientBounds2D(r, p, f, φ)
    αs = [Vector(2.0.^(-7:0.5:2)); 2^(2)]
    restart = 150 # terminate GMRES solver after 150 iterations
    tic = @elapsed u, ψ, iters = pg_hierarchical_solve(PG, αs,
            matrixfree=true,backtracking=false,its_max=4,show_trace=show_trace,
            gmres_baseline_tol=gmres_tol,
            c_1=-1e4,pf_its_max=4, β=β, restart=restart)
    return u, tic, iters
end

"""
p-uniform refinement, fixed h
"""
# Solve and count the GMRES iterations
tics, newton_its, gmres_its = T[], Int32[], Int32[]
for p in [5,10,15,20]
    N = 2
    print("p=$p, mesh level: $N.\n")
    r = range(0,1,2^(N+1)+1)
    u, tic, iters = gradient_bound_solve(r, p, f, φ, gmres_tol=1e-3, β=1/10^(N+p+1), show_trace=true);
    push!(newton_its, iters[1])
    push!(gmres_its, iters[2])
end
avg_its = gmres_its ./ newton_its
save_data([], [], newton_its, avg_its, "p_uniform")

# Solve again with show_trace = false for wall-clock timings
for p in [5,10,15,20]
    N = 2
    print("p=$p, mesh level: $N.\n")
    r = range(0,1,2^(N+1)+1)
    u, tic, iters = gradient_bound_solve(r, p, f, φ, gmres_tol=1e-5, β=1e-6, show_trace=false);
    push!(tics, tic)
end
avg_tics = tics ./ newton_its
save_data(tics, avg_tics, newton_its, avg_its, "p_uniform")


"""
h-uniform refinement, fixed p
"""
for p in [1,3]
    tics, newton_its, gmres_its = T[], Int32[], Int32[]
    n = p == 1 ? 3 : 2
    # Solve and count the GMRES iterations
    for N in [n,n+2,n+4]
        print("p=$p, mesh level: $N.\n")
        r = range(0,1,2^(N+1)+1)
        u, tic, iters = gradient_bound_solve(r, p, f, φ, gmres_tol=1e-3, β=1/10^(N+p+1), show_trace=true);
        push!(newton_its, iters[1])
        push!(gmres_its, iters[2])
    end
    avg_its = gmres_its ./ newton_its
    save_data([], [], newton_its, avg_its, "h_uniform_p_$p")

    # Solve again with show_trace = false for wall-clock timings
    for N in [n,n+2,n+4]
        print("p=$p, mesh level: $N.\n")
        r = range(0,1,2^(N+1)+1)
        u, tic, iters = gradient_bound_solve(r, p, f, φ, gmres_tol=1e-3, β=1/10^(N+p+1), show_trace=false);
        push!(tics, tic)
    end
    avg_tics = tics ./ newton_its
    save_data(tics, avg_tics, newton_its, avg_its, "h_uniform_p_$p")
end

"""
hp-uniform refinement
"""
tics, newton_its, gmres_its = T[], Int32[], Int32[]
# Solve and count the GMRES iterations
for N in [3,5]
    p = N
    print("p=$p, mesh level: $N.\n")
    r = range(0,1,2^(N+1)+1)
    u, tic, iters = gradient_bound_solve(r, p, f, φ, gmres_tol=1e-3, β=1/10^(N+p+1), show_trace=true);
    push!(newton_its, iters[1])
    push!(gmres_its, iters[2])
end
avg_its = gmres_its ./ newton_its
save_data([], [], newton_its, avg_its, "hp_uniform")

# Solve again with show_trace = false for wall-clock timings
for N in [3,5]
    p = N
    print("p=$p, mesh level: $N.\n")
    r = range(0,1,2^(N+1)+1)
    u, tic, iters = gradient_bound_solve(r, p, f, φ, gmres_tol=1e-3, β=1/10^(N+p+1), show_trace=false);
    push!(tics, tic)
end
avg_tics = tics ./ newton_its
save_data(tics, avg_tics, newton_its, avg_its, "hp_uniform")


#### Solution check
if false
    u_ref = readdlm(path*"../reference/u_ref_h_uniform.log")[:,1]
    Dp = DirichletPolynomial(range(0,1,2^7+1))
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

    N=1;PG = GradientBounds2D(range(0,1,2^(N+1)+1), N, f, φ)
    approximate_error(PG, u, Dp, 6, u_ref)
end