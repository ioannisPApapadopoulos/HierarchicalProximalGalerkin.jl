using HierarchicalProximalGalerkin, LinearAlgebra
# using Plots, LaTeXStrings
using DelimitedFiles

path = "output/thermoforming/"
if !isdir(path)
    mkpath(path)
end
function save_data(alg_its, its1, its2, h1s, subpath)
    stepI = [its1[1] its1[2] its1[1]/alg_its its1[2]/its1[1]]
    stepII = [its2[1] its2[2] its2[1]/alg_its its2[2]/its1[1]]
    writedlm(path*subpath*"_fixed_point.log", alg_its)
    writedlm(path*subpath*"_step_I.log", stepI)
    writedlm(path*subpath*"_step_II.log", stepII)
    writedlm(path*subpath*"_h1s.log",  h1s)
end

# Model parameters
k = 1.0
Φ₀(x,y) = 1.1 - 2*max(abs(x -1/2),  abs(y-1/2)) +  0.1*cos(8*π*x)*cos(8*π*y)
ϕ(x,y) = sin(π*x)*sin(π*y)
f(x,y) = 100.0
function g(s)
    if s ≤ 0
        return 1/5
    elseif 0 < s < 1
        return (1-s)/5
    else
        return 0.0
    end
end
function dg(s)
    if s ≤ 0
        return 0.0
    elseif 0 < s < 1
        return -1/5
    else
        return 0.0
    end
end


function obstacle(x,y,p,T,Φ₀,ϕ,C)
    rT = reshape(T,isqrt(lastindex(T)), isqrt(lastindex(T)))
    Φ₀(x,y) + ϕ(x,y)*(C[y,Block.(1:p+1)]' * rT * C[x,Block.(1:p+1)])
end

r = range(0,1,5)
C = ContinuousPolynomial{1}(r)
its1, its2, alg_its, tics = [], [], [], []
h1s = Float64[]

for p in 11:11
    n = length(r)-1
    TF = Thermoforming2D(Vector(r), p, p, k, Φ₀, ϕ, g, dg);

    global T = zeros((n*(p+1)+1)^2)
    global u = zeros((n*(p+1)-1)^2)

    ob_its = (0,0)
    Phi_its = (0,0)
    iter2 = 0

    Dp = DirichletPolynomial(r)
    Δ = weaklaplacian(Dp)
    KR = Block.(oneto(p+1))
    A1 = sparse(Symmetric(-parent(Δ)[KR,KR]))
    M1 = sparse(Symmetric((Dp' * Dp)[KR,KR]))
    A = sparse(kron(A1,M1) + kron(M1,A1))
    M = sparse(kron(M1,M1))


    for iter in 1:10
        print("Fixed point iteration: $iter.\n")
        ob(x,y) = obstacle(x,y,p,T,Φ₀,ϕ,C)
        PG = ObstacleProblem2D(r, p, f, ob)#, b=p_u);

        αs = 2.0.^(-6:2:0)

        u_, ψ, iters = pg_hierarchical_solve(PG, αs, gmres_baseline_tol=1e-7, gmres_abstol=1e-7, 
                c_1=-1e4,
                matrixfree=true,backtracking=true,show_trace=true,
                restart=min(size(PG.A, 1), 300), its_max=4, pf_its_max=4, β=0.0, tolerance=1e-6)

        ob_its = ob_its .+ iters

        # print("\n\n")
        global T, its = HierarchicalProximalGalerkin.solve(TF, u_, T0=T, show_trace=false,matrixfree=true)
        Phi_its = Phi_its .+ its

        d = u-u_
        l2 = sqrt(d' * M * d)
        h1 = sqrt(d' * A * d + l2^2)
        push!(h1s, h1)
        print("‖u_n+1 - u_n‖_L^2 = $l2, ‖u_n+1 - u_n‖_H^1 = $h1.\n")
        iter2 += 1

        global u = copy(u_)
        if h1 < 1e-2
            print("Convergence reached.\n")
            break
        end
    end
    push!(its1, ob_its); push!(its2, Phi_its); push!(alg_its, iter2);
    save_data(alg_its, its1, its2, h1s, "p_$(p+1)")
end

# Dp = DirichletPolynomial(r)
# xx = range(0,1,500)
# Ux = evaluate2D(u, xx, xx, 16, Dp)
# # norm(U[findall(U .> 1.0)] .- 1.0, Inf)
# Plots.gr_cbar_offsets[] = (-0.05,-0.01)
# Plots.gr_cbar_width[] = 0.03
# surface(xx,xx,Ux,
#     color=:diverging, #:vik,
#     xlabel=L"x", ylabel=L"y", zlabel=L"u(x,y)",
#     # camera=(30,-30),
#     title="Membrane  "*L"u",
#     margin=(-6, :mm),
#     # zlim=[0,1.3],
# )
# Plots.savefig("thermoforming-membrane.pdf")

# ob(x,y) = obstacle(x,y,15,T,Φ₀,ϕ,C)
# surface(xx,xx,ob.(xx',xx),
#     color=:diverging, #:vik,
#     xlabel=L"x", ylabel=L"y", zlabel=L"(\Phi_0 + \xi T)(x,y)",
#     # camera=(30,-30),
#     title="Mould  "*L"\Phi_0 + \xi T",
#     margin=(-6, :mm),
#     # right_margin=3Plots.mm
#     # extra_kwargs=Dict(:subplot=>Dict("3d_colorbar_axis" => [0.1, 0.1, 0.1, 0.1]) )
#     # zlim=[0,1.3],
# )
# Plots.savefig("thermoforming-mould.pdf")

# y = 0.5
# p = 15
# xx = range(0,1,500)
# ux = evaluate2D(u, xx, [y], p+1, Dp)'
# ob(x,y) = obstacle(x,y,p,T,Φ₀,ϕ,C)
# ox = ob.(xx, y)
# Tx = evaluate2D(T, xx, [y], p+1, C)'
# origx = Φ₀.(xx,y)

# Plots.plot(xx, [ux ox Tx origx],
#     linewidth=2,
#     label=["Membrane" "Mould" "Temperature" "Original Mould"],
#     linestyle=[:solid :dash],
#     xlabel=L"x",
#     title=L"Slice at $y=1/2$",
#     xlabelfontsize=20,
#     # ylim=[0,1.3]
# )
# # Plots.savefig("thermoforming-slice.pdf")