using HierarchicalProximalGalerkin
using SparseArrays, LinearAlgebra
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
    bf_p=p+10;
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

function benchmark_solve(r::AbstractVector{T}, p::Union{<:Int, Vector{<:Int}}, refinement_strategy::Function; nlevels::Int=6) where T
    gmres_its, newton_its, tics = Int32[], Int32[], T[]
    rs, ps = Vector{T}[], eltype(p)[]
    ndofs = Int64[]
    l2s, h1s = T[], T[]
    PG, u, ψ, w = [], [], [], []
    for iter in 1:nlevels

        print("Considering mesh refinement $iter.\n")
        PG = BCsObstacleProblem2D(r, p, f, φ, ue);
        Md = sparse(grammatrix(PG.Dp)[Block.(oneto(p+1)), Block.(oneto(p+1))])
        Md = kron(Md, Md)

        push!(ndofs, lastindex(PG.A,1))
        push!(rs, r)
        push!(ps, p)
        
        αs = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e3, 1e3]
        tic = @elapsed u, ψ, w, iters = pg_hierarchical_solve(PG, αs,
            matrixfree=true,backtracking=true,its_max=30,pf_its_max=5,return_w=true,show_trace=true, c_1=-1e4, Md=Md)


        push!(tics, tic)
        push!(gmres_its, iters[2])
        push!(newton_its, iters[1])

        l2, h1 = compute_errors(PG, u, p)
        push!(l2s, l2)
        push!(h1s, h1)

        MA = MeshAdaptivity2D(PG, αs[end], u, ψ, w, f, φ, bcs=ue);
        r, p = refinement_strategy(r, p, MA)

    end
    return (l2s, h1s), ndofs, (newton_its,gmres_its), rs, ps, tics, (PG, u, ψ, w)
end


function pg_h_uniform_refine(r::AbstractVector, p::Int, MA)
    uniform_refine(r), p
end
(l2s_u, h1s_u), ndofs, iters, rs, ps, tics, (PG, u, ψ, w) = 
    benchmark_solve(range(-1.5,1.5,11), 2, pg_h_uniform_refine, nlevels=4)
Plots.plot(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)

function pg_h_adaptive_refine(r::AbstractVector, p::Int, MA)
    ϵs = error_estimates(MA)
    h_refine(MA,ϵs,δ=0.5), p
end
(l2s_u, h1s_u), ndofs, iters, rs, ps, tics, (PG, u, ψ, w) = 
    benchmark_solve(range(-1.5,1.5,11), 1, pg_h_adaptive_refine, nlevels=8)
Plots.plot!(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)
Plots.plot(rs[end], ones(length(rs[end])), marker=:dot)

function pg_h_adaptive_p_uniform_refine(r::AbstractVector, p::Int, MA)
    ϵs = error_estimates(MA)
    h_refine(MA,ϵs,δ=0.5), p+2
end
(l2s_u, h1s_u), ndofs, iters, rs, ps, tics, (PG, u, ψ, w) = 
    benchmark_solve(range(-1.5,1.5,7), 1, pg_h_adaptive_p_uniform_refine, nlevels=4);
Plots.plot(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)

function pg_p_uniform_refine(r::AbstractVector, p::Int, MA)
    r, p+1
end
(l2s_u, h1s_u), ndofs, iters, rs, ps, tics, (PG, u, ψ, w) = 
    benchmark_solve(range(-1.5,1.5,11), 1, pg_p_uniform_refine, nlevels=10)
Plots.plot!(ndofs, h1s_u, marker=:xcross, linewidth=2, xaxis=:log10, yaxis=:log10)

MA = MeshAdaptivity2D(PG, 1e3, u, ψ, w, f, φ, bcs=ue);
ϵs = error_estimates(MA)
xx = (rs[end][1:end-1]+rs[end][2:end]) ./ 2
contourf(xx,xx,ϵs,
           color=:diverging,
           xlabel=L"x",ylabel=L"y", zlabel=L"u(x,y)", aspect_ratio=:equal)

xx = range(-1.5,1.5,101)
Ux = evaluate2D(u, xx, xx, PG.p+1, PG.C)
Ua = ue.(xx, xx')

heatmap(xx,xx,abs.(Ux-Ua),
           color=cgrad(:diverging, scale = :log10),
           xlabel=L"x",ylabel=L"y", zlabel=L"u(x,y)", aspect_ratio=:equal, 
           grid=false, xlim=[-1.5,1.5], ylim=[-1.5,1.5])
vline!(rs[end], color=:black, linewidth=0.8, label="")
hline!(rs[end], color=:black, linewidth=0.8, label="")


p = Plots.plot(ndofs_p, h1s_p, 
    yaxis=:log10, xaxis=:log10,  
    marker=:dtriangle,
    label="(S1)",
    ylabel=L"\Vert u - u_{hp} \Vert_{H^1(\Omega)}", xlabel="Degrees of freedom",
    ylabelfontsize=15,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=10)
ps = 1:7
Plots.plot!(ndofs_p, [1^(1.5) ./ ps.^(1.5) * h1s_p[1]], linestyle=:dash, label=L"$O(p^{-3/2}), O(h^3/2)$")

xx = range(-1.5,1.5,101)
Plots.gr_cbar_offsets[] = (-0.05,-0.01)
Plots.gr_cbar_width[] = 0.03
Ux = evaluate2D(uc, xx, xx, p+1, PG.C)
surface(xx,xx,Ux,
    color=:diverging,
    margin=(-6, :mm),
    topmargin=(1, :mm),
    xlabel=L"x",ylabel=L"y", zlabel=L"u(x,y)")
Plots.savefig("benchmark.pdf")