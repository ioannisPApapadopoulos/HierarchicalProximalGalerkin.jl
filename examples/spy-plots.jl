using HierarchicalProximalGalerkin, SparseArrays, MatrixFactorizations
using CuthillMcKee, IterativeSolvers
using Plots, LaTeXStrings

## 1D
f(x) = 1.0
φ(x) = 1.0
f(x,y) = 1.0
φ(x,y) = 1.0

r, p = range(0,1,6), 10
OP = ObstacleProblem(r, p, f, φ);

A, B, rcholA = OP.A, OP.B, OP.rchol_A

A[findall(abs.(A) .> 1e-15)] .= 1.0;
Plots.spy(A, markersize=2, size=(400,400), aspect_ratio=:equal)
Plots.savefig("A1d.pdf")

L = Matrix(rcholA.L);
L[findall(abs.(L) .> 1e-15)] .= 1.0;
Plots.spy(L, markersize=2, size=(400,400), aspect_ratio=:equal)
Plots.savefig("rcholA1d.pdf")

B[findall(abs.(B) .> 1e-15)] .= 1.0;
Plots.spy(B, markersize=2, size=(400,400), aspect_ratio=:equal)
Plots.savefig("B1d.pdf")

ψ = rand(size(B,2))
D = assemble_D(OP, ψ);
D[findall(abs.(D) .> 1e-15)] .= 1.0;
Plots.spy(D, markersize=2, size=(400,400), aspect_ratio=:equal)
Plots.savefig("D1d.pdf")

rD = rcmpermute(sparse(D))
Plots.spy(rD, markersize=2, size=(400,400), aspect_ratio=:equal)
Plots.savefig("rD1d.pdf")

### 2D


r, p = range(0,1,6), 5
OP = ObstacleProblem2D(r, p, f, φ);

A, B, rcholA = sparse(OP.A), sparse(OP.B), OP.rchol_A

A[findall(abs.(A) .> 1e-15)] .= 1.0;
Plots.spy(A, markersize=1, size=(400,400), aspect_ratio=:equal)
Plots.savefig("A2d.pdf")


L = Matrix(sparse(rcholA.L));
L[findall(abs.(L) .> 1e-15)] .= 1.0;
Plots.spy(L, markersize=1, size=(400,400), aspect_ratio=:equal)
Plots.savefig("cholA2d.pdf")

B[findall(abs.(B) .> 1e-15)] .= 1.0;
Plots.spy(B, markersize=1, size=(400,400), aspect_ratio=:equal)
Plots.savefig("B2d.pdf")

ψ = rand(size(B,2))
D = assemble_D(OP, ψ);
D[findall(abs.(D) .> 1e-15)] .= 1.0;
Plots.spy(D, markersize=1, size=(400,400), aspect_ratio=:equal)
Plots.savefig("D2d.pdf")


rD = rcmpermute(D)

@time MatrixFactorizations.lu(D);
@time MatrixFactorizations.lu(rD);
# rD = zeros(size(D))
# K = length(r)-1
# for pc in 1:K
#     rD[p*(pc-1)+1:p*pc,p*(pc-1)+1:p*pc] = D[pc:K:end, pc:K:end]
# end
Plots.spy(rD, markersize=1, size=(400,400), aspect_ratio=:equal)
Plots.savefig("rD2d.pdf")



## Schur complement condition number

## 1D

r = range(0,1,11)
iters_p, tics_p = Int[], Float64[]
for p in 1:40
    PG = ObstacleProblem(r,p,f,φ);
    S = PG.B' * (PG.A \ Matrix(PG.B))
    Ŝ = PG.E
    tic = @elapsed lu_E = MatrixFactorizations.lu(Ŝ)
    push!(tics_p, tic)
    b = ones(size(S,1))
    _, info = IterativeSolvers.gmres(S, b, Pl=lu_E, restart=size(S,1), reltol=1e-6, abstol=1e-14, log=true)
    push!(iters_p, info.iters)
end

# pred_p = iters_p[1] * 1.5 *( 1 .+ log.(2:21).^2)
Plots.plot(2:41,[iters_p], marker=[:circle :none],
    linestyle = [:solid :dash],
    linewidth=2,
    legend=:bottomright,
    label="p-refinement",
    ylabel="GMRES iterations",
    xlabel=L"$p$",
    xlabelfontsize=20,ylabelfontsize=17,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("p-gmres-1d.pdf")

Plots.plot(2:41, tics_p,
    legend=:none,
    linewidth=2,
    marker=:circle,
    ylabel="LU factorization time (s)",
    xlabel=L"$p$",
    yscale=:log10,
    xlabelfontsize=20,ylabelfontsize=15,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("p-factorization-time-1d.pdf")

p=4
conds_h = Float64[]
iters_h, tics_h = Int[],  Float64[]
for n in 2 .^(1:9)
    PG = ObstacleProblem(range(0,1,n+1),p,f,φ);
    S = PG.B' * (PG.A \ Matrix(PG.B))
    Ŝ = PG.E
    tic = @elapsed lu_E = MatrixFactorizations.lu(Ŝ)
    push!(tics_h, tic)
    b = ones(size(S,1))
    _, info = IterativeSolvers.gmres(S, b, Pl=lu_E, restart=size(S,1), reltol=1e-6, abstol=1e-14, log=true)
    push!(iters_h, info.iters)
end

ns = 2 .^(1:9)
pred_h =  0.75*iters_h[1] * (1 .+ log.(ns).^2)# * iters_h[1] #./ log(2)^2
Plots.plot(2 .^(1:9),[iters_h pred_h], marker=[:circle :none],
    linestyle = [:solid :dash],
    linewidth=2,
    legend=:topleft,
    label=["h-refinement" L"$O((\mathrm{log} \, h^{-1})^2)$"],
    ylabel="GMRES iterations",
    xlabel=L"$1/h$",
    xlabelfontsize=20,ylabelfontsize=17,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("h-gmres-1d.pdf")

Plots.plot(ns, tics_h,
    legend=:none,
    linewidth=2,
    marker=:circle,
    yscale=:log10,
    ylabel="LU factorization time (s)",
    xlabel=L"$1/h$",
    xlabelfontsize=20,ylabelfontsize=15,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("h-factorization-time-1d.pdf")


## 2D
r = range(0,1,5)
iters_p, tics_p = Int[], Float64[]
for p in 1:20
    PG = ObstacleProblem2D(r,p,f,φ);
    S = PG.B' * (PG.A \ Matrix(PG.B))
    Ŝ = PG.E
    tic = @elapsed lu_E = MatrixFactorizations.lu(Ŝ)
    push!(tics_p, tic)
    b = ones(size(S,1))
    _, info = IterativeSolvers.gmres(S, b, Pl=lu_E, restart=size(S,1), reltol=1e-6, abstol=1e-14, log=true)
    push!(iters_p, info.iters)
end

pred_p = iters_p[1] * 1.6 *( log.(2:21).^2)
Plots.plot(2:21,[iters_p pred_p], marker=[:circle :none],
    linestyle = [:solid :dash],
    linewidth=2,
    legend=:topleft,
    label=["p-refinement" L"$O(\mathrm{log}^{2} p)$"],
    ylabel="GMRES iterations",
    xlabel=L"$p$",
    xlabelfontsize=20,ylabelfontsize=17,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("p-gmres.pdf")

Plots.plot(2:21, tics_p,
    legend=:none,
    linewidth=2,
    marker=:circle,
    ylabel="LU factorization time (s)",
    xlabel=L"$p$",
    yscale=:log10,
    xlabelfontsize=20,ylabelfontsize=15,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("p-factorization-time.pdf")

p=4
conds_h = Float64[]
iters_h, tics_h = Int[],  Float64[]
for n in [2,4,8,16,32]
    PG = ObstacleProblem2D(range(0,1,n+1),p,f,φ);
    S = PG.B' * (PG.A \ Matrix(PG.B))
    Ŝ = PG.E
    tic = @elapsed lu_E = MatrixFactorizations.lu(Ŝ)
    push!(tics_h, tic)
    b = ones(size(S,1))
    _, info = IterativeSolvers.gmres(S, b, Pl=lu_E, restart=size(S,1), reltol=1e-6, abstol=1e-14, log=true)
    push!(iters_h, info.iters)
end

ns = [2,4,8,16,32]
pred_h =  0.75*iters_h[1] * (1 .+ log.(ns).^2)# * iters_h[1] #./ log(2)^2
Plots.plot([2,4,8,16,32],[iters_h pred_h], marker=[:circle :none],
    linestyle = [:solid :dash],
    linewidth=2,
    legend=:topleft,
    label=["h-refinement" L"$O((\mathrm{log} \, h^{-1})^2)$"],
    ylabel="GMRES iterations",
    xlabel=L"$1/h$",
    xlabelfontsize=20,ylabelfontsize=17,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("h-gmres.pdf")

Plots.plot(ns, tics_h,
    legend=:none,
    linewidth=2,
    marker=:circle,
    yscale=:log10,
    ylabel="LU factorization time (s)",
    xlabel=L"$1/h$",
    xlabelfontsize=20,ylabelfontsize=15,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("h-factorization-time.pdf")

## Gradient bounds

β = 1e-5
r = range(0,1,5)
iters_p, tics_p = Int[], Float64[]
p = 20
for p in 1:20
    PG = GradientBounds2D(r,p,f,φ);
    E = blockdiag(PG.E,PG.E)

    # S = blockdiag(PG.M,PG.M) + PG.B' * (PG.A \ Matrix(PG.B))
    # Ŝ = blockdiag(PG.M,PG.M) + PG.E # 1e-10*PG.M + PG.E# + 1e-5*PG.M
    S = β*E + PG.B' * (PG.A \ Matrix(PG.B))
    Ŝ = β*E # 1e-10*PG.M + PG.E# + 1e-5*PG.M
    tic = @elapsed lu_E = MatrixFactorizations.lu(Ŝ)
    push!(tics_p, tic)
    b = -ones(size(S,1))
    _, info = IterativeSolvers.gmres(S, b, Pl=lu_E, restart=size(S,1), reltol=1e-6, abstol=1e-14, log=true)
    push!(iters_p, info.iters)
end

pred_p =  ( 2e1*log.(1:20)) # 6* iters_p[1] * (2:21) .- 11
Plots.plot(1:20,[iters_p pred_p], marker=[:circle :none],
    linestyle = [:solid :dash],
    linewidth=2,
    legend=:bottomright,
    label=["p-refinement" L"$O(\mathrm{log} p)$"],
    ylabel="GMRES iterations",
    xlabel=L"$p$",
    xlabelfontsize=20,ylabelfontsize=17,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("p-gmres-gradient.pdf")

Plots.plot(1:20, tics_p,
    legend=:none,
    linewidth=2,
    marker=:circle,
    ylabel="LU factorization time (s)",
    xlabel=L"$p$",
    yscale=:log10,
    xlabelfontsize=20,ylabelfontsize=15,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("p-factorization-time-gradient.pdf")

p=4
conds_h = Float64[]
iters_h, tics_h = Int[],  Float64[]
for n in [2,4,8,16]
    PG = GradientBounds2D(range(0,1,n+1),p,f,φ);
    E = blockdiag(PG.E,PG.E)
    S = β*E + PG.B' * (PG.A \ Matrix(PG.B))
    Ŝ = β*E
    tic = @elapsed lu_E = MatrixFactorizations.lu(Ŝ)
    push!(tics_h, tic)
    b = -ones(size(S,1))
    _, info = IterativeSolvers.gmres(S, b, Pl=lu_E, restart=size(S,1), reltol=1e-6, abstol=1e-14, log=true)
    push!(iters_h, info.iters)
end

ns = [2,4,8,16]
pred_h =  iters_h[1] * (log.(ns)) ./ log(ns[1])# * iters_h[1] #./ log(2)^2
Plots.plot([2,4,8,16],[iters_h pred_h], marker=[:circle :none],
    linestyle = [:solid :dash],
    linewidth=2,
    legend=:topleft,
    label=["h-refinement" L"$O(\mathrm{log} \, h^{-1})$"],
    ylabel="GMRES iterations",
    xlabel=L"$1/h$",
    xlabelfontsize=20,ylabelfontsize=17,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("h-gmres-gradient.pdf")

Plots.plot(ns, tics_h,
    legend=:none,
    linewidth=2,
    marker=:circle,
    yscale=:log10,
    ylabel="LU factorization time (s)",
    xlabel=L"$1/h$",
    xlabelfontsize=20,ylabelfontsize=15,xtickfontsize=15,ytickfontsize=15,legendfontsize=15)
Plots.savefig("h-factorization-time-gradient.pdf")