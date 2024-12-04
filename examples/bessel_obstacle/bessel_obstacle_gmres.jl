using MKL
using HierarchicalProximalGalerkin
using IterativeSolvers, LinearAlgebra
using Plots, LaTeXStrings, DelimitedFiles
using SpecialFunctions

"""
This example requires the package ```SpecialFunctions.jl```

Section 7.2: Solve a 2D obstacle problem with a bessel-type oscillatory obstacle.

Here we use the hpG solver with a preconditioned GMRES solver via
    (i)   p-uniform refinement
    (ii)  h-uniform refinement with p=2, 3
    (iii) hp-uniform refinement

We run each solve twice. Once to obtain GMRES iteration counts via
show_trace=true and again to obtain the true wall-clock timings.
"""

T = Float64
f(x,y) = 100.0
φ(x,y) = (besselj(0,20x)+1)*(besselj(0,20y)+1)

path = "output/bessel_obstacle/gmres/"
if !isdir(path)
    mkpath(path)
end
function save_data(tics, avg_tics, its, subpath)
    writedlm(path*subpath*"_avg_tics.log", avg_tics)
    writedlm(path*subpath*"_tics.log", tics)
    writedlm(path*subpath*"_avg_gmres_its.log", its)
end

function bessel_solve(r::AbstractVector{T}, p::Int, f::Function, φ::Function; its_max::Int=6, show_trace::Bool=false) where T
    PG = ObstacleProblem2D(r, p, f, φ)
    αs = [Vector(2.0.^(-7:0.5:-3)); 2^(-3)]

    restart = 100 # terminate GMRES solver after 100 iterations
    tic = @elapsed u, ψ, iters = pg_hierarchical_solve(PG, αs,
        matrixfree=true,backtracking=false,its_max=its_max,show_trace=show_trace,
        gmres_baseline_tol=1e-5,
        c_1=-1e4,pf_its_max=2, β=1e-4, restart=restart)
    return u, tic, iters
end


"""
p-uniform refinement, fixed h
"""
# Solve and count the GMRES iterations
tics, newton_its, gmres_its = T[], Int32[], Int32[]
for p in [2, 9, 16, 23]
    N = 1
    print("p=$p, mesh level: $N.\n")
    r = range(0,1,10*2^(N-1)+1)
    u, tic, iters = bessel_solve(r, p, f, φ, show_trace=true);
    push!(newton_its, iters[1])
    push!(gmres_its, iters[2])
end
avg_its = gmres_its ./ newton_its

# Solve again with show_trace = false for wall-clock timings
for p in [2, 9, 16, 23]
    N = 1
    print("p=$p, mesh level: $N.\n")
    r = range(0,1,10*2^(N-1)+1)
    u, tic, iters = bessel_solve(r, p, f, φ, show_trace=false);
    push!(tics, tic)
end
avg_tics = tics ./ newton_its
save_data(tics, avg_tics, avg_its, "p_uniform")

"""
h-uniform refinement, fixed p
"""
for p in [1,2]
    tics, newton_its, gmres_its = T[], Int32[], Int32[]
    Ns = p == 2 ? [1,3,5] : [2,4,6]
    # Solve and count the GMRES iterations
    for N in Ns
        print("p=$p, mesh level: $N.\n")
        r = range(0,1,10*2^(N-1)+1)
        u, tic, iters = bessel_solve(r, p, f, φ, show_trace=true);
        push!(newton_its, iters[1])
        push!(gmres_its, iters[2])
    end
    avg_its = gmres_its ./ newton_its
    # Solve again with show_trace = false for wall-clock timings
    for N in Ns
        print("p=$p, mesh level: $N.\n")
        r = range(0,1,10*2^(N-1)+1)
        u, tic, iters = bessel_solve(r, p, f, φ, show_trace=false);
        push!(tics, tic)
    end
    avg_tics = tics ./ newton_its
    save_data(tics, avg_tics, avg_its, "h_uniform_p_$p")
end

"""
hp-uniform refinement
"""
tics, newton_its, gmres_its = T[], Int32[], Int32[]
# Solve and count the GMRES iterations
for N in [1,3,5]
    p = N
    print("p=$p, mesh level: $N.\n")
    r = range(0,1,10*2^(N-1)+1)
    u, tic, iters = bessel_solve(r, p, f, φ, show_trace=true);
    push!(newton_its, iters[1])
    push!(gmres_its, iters[2])
end
avg_its = gmres_its ./ newton_its
# Solve again with show_trace = false for wall-clock timings
for N in [1,3,5]
    p = N
    print("p=$p, mesh level: $N.\n")
    r = range(0,1,10*2^(N-1)+1)
    u, tic, iters = bessel_solve(r, p, f, φ, show_trace=false);
    push!(tics, tic)
end
avg_tics = tics ./ newton_its
save_data(tics, avg_tics, avg_its, "hp_uniform")



#### Check error in solutions
if false
    u6 = Vector(readdlm(path*"../hp-uniform-u_6.log")[:,1])
    p=6
    Dp  = DirichletPolynomial(range(0,1,10*2^(6-1)+1))
    u_ref = reshape(u6, isqrt(lastindex(u6)), isqrt(lastindex(u6)))

    KR = Block.(oneto(p+1))
    M1 = sparse(Symmetric((Dp' * Dp)[KR,KR]))
    M = sparse(Symmetric(kron(M1,M1)))

    Δ = weaklaplacian(Dp)
    A1 = sparse(Symmetric(-parent(Δ)[KR,KR]))
    A = sparse(kron(A1,M1) + kron(M1,A1))

    xy, plan_D = plan_grid_transform(Dp, Block(p+1,p+1));
    x,y=first(xy),last(xy);
    l2s, h1s = T[], T[]

    N=1;PG = ObstacleProblem2D(range(0,1,10*2^(N-1)+1), 9, f, φ)
    vals = evaluate2D(us[end], x, y, PG.p+1, PG.Dp);
    d = (u_ref - plan_D * vals)[:]
    push!(l2s, sqrt(d' * (M * d)))
    push!(h1s, sqrt(d' * (A * d) + l2s[end]^2))
end
