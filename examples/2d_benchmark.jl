using HierarchicalProximalGalerkin
using SparseArrays
using Plots, LaTeXStrings
using BlockArrays

function ue(x,y)
    r2 = x^2 + y^2
    if r2 ≥ 1
        return log(sqrt(r2)) - r2/2 + 3/2
    else
        return 1.0
    end
end

f(x,y) = 2.0
φ(x,y) = 1.0

function compute_errors(PG::BCsObstacleProblem2D{T}, uc::Vector{T}, p::Int64) where T
    bf_p=p+2;
    (xg,yg), plan_C = plan_grid_transform(PG.C, Block(PG.p+bf_p,PG.p+bf_p));
    uec = (plan_C * (ue.(xg,reshape(yg,1,1,size(yg)...))))[:]
    KR = Block.(oneto(PG.p+bf_p))
    M1 = sparse(grammatrix(PG.C)[KR, KR]);
    M = kron(M1,M1);
    Δ = weaklaplacian(PG.C)
    A1 = sparse(Symmetric(-parent(Δ)[KR,KR]))
    A = sparse(kron(A1,M1) + kron(M1,A1))

    n = PG.Nh
    u1 = reshape(uc, (n*(p+1))+1, (n*(p+1))+1)
    u1 = zero_pad(u1,  (n*(p+bf_p))+1)
    d = uec-u1[:]
    l2 =  sqrt(d' * M * d)
    h1 = sqrt(d' * A * d + l2^2)
    l2, h1
end

T = Float64
gmres_its_h, newton_its_h, tics_h = Int32[], Int32[], T[]
rs_h = Vector{T}[]
ndofs_h = Int64[]
l2s_h, h1s_h = T[], T[]
global r = range(-1.5,1.5,11)
global p = 1
for iter in 1:1 # 4

    print("Considering mesh refinement $iter.\n")
    PG = BCsObstacleProblem2D(r, p, f, φ, ue);

    push!(ndofs_h, lastindex(PG.A,1))
    push!(rs_h, r)
    
    αs = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e3, 1e3]
    tic = @elapsed uc, ψ, w, iters = pg_hierarchical_solve(PG, αs,
        matrixfree=true,backtracking=true,its_max=30, return_w=true,show_trace=true)


    push!(tics_h, tic)
    push!(gmres_its_h, iters[2])
    push!(newton_its_h, iters[1])

    l2, h1 = compute_errors(PG, uc, p)
    push!(l2s_h, l2)
    push!(h1s_h, h1)

    # MA = MeshAdaptivity2D(PG, 1e3, uc, ψ, w, f, φ, bcs=ue);
    # ϵs = error_estimates(MA)
    # global r = h_refine(MA,ϵs,δ=0.01)
    global r = uniform_refine(r)

end


gmres_its_p, newton_its_p, tics_p = Int32[], Int32[], T[]
rs_p = Vector{T}[]
ndofs_p = Int64[]
l2s_p, h1s_p = T[], T[]
r = range(-1.5,1.5,11)
for p in 2:8
    PG = BCsObstacleProblem2D(r, p, f, φ, ue);
    push!(ndofs_p, lastindex(PG.A,1))
    push!(rs_p, r)
    
    αs = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e3, 1e3]
    tic = @elapsed uc, ψ, w, iters = pg_hierarchical_solve(PG, αs, β=1e-8,
        matrixfree=true,backtracking=true,its_max=30, return_w=true,show_trace=false)

    push!(tics_p, tic)
    push!(gmres_its_p, iters[2])
    push!(newton_its_p, iters[1])

    l2, h1 = compute_errors(PG, uc, p)
    push!(l2s_p, l2)
    push!(h1s_p, h1)

    # MA = MeshAdaptivity2D(PG, 1e3, uc, ψ, w, f, φ, bcs=ue);
    # ϵs = error_estimates(MA)
    # r = h_refine(MA,ϵs,δ=0.01)
    # r = uniform_refine(r)
    # p+=1


end

p = Plots.plot(ndofs_p, h1s_p, 
    yaxis=:log10, xaxis=:log10,  
    marker=:dtriangle,
    label="(S1)",
    ylabel=L"\Vert u - u_{hp} \Vert_{H^1(\Omega)}", xlabel="Degrees of freedom",
    ylabelfontsize=15,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=10)
ps = 1:7
Plots.plot!(ndofs_p, [1^(1.5) ./ ps.^(1.5) * h1s_p[1]], linestyle=:dash, label=L"$O(p^{-3/2}), O(h^3/2)$")

xx = range(-1.5,1.5,100)
Plots.gr_cbar_offsets[] = (-0.05,-0.01)
Plots.gr_cbar_width[] = 0.03
Ux = evaluate2D(uc, xx, xx, p+1, PG.C)
surface(xx,xx,Ux,
    color=:redsblues,
    margin=(-6, :mm),
    topmargin=(1, :mm),
    xlabel=L"x",ylabel=L"y", zlabel=L"u(x,y)")
Plots.savefig("benchmark.pdf")