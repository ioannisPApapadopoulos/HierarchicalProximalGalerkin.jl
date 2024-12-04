using DelimitedFiles
using Plots, LaTeXStrings

"""
Section 7.3

Helper script for plotting convergence of gradient-type constraint problem
"""

path = "output/gradient-bound/"

markers = [:circle, :dtriangle, :rect, :diamond,:star4, :hexagon, :cross, :utriangle,  :xcross, 
    :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon,  :star6, 
    :star7, :star8, :vline, :hline, :+, :x]

ndofs, h1s, avg_tics, tics = [], [], [], []
strs = path .* [
                "p_uniform",
                "h_uniform_p_1",
                "h_uniform_p_2",
                "h_uniform_p_3",
                # "h_uniform_p_5",
                "hp_uniform"
]

for str in strs[1:end]
    push!(h1s, readdlm(str*"_h1s.log"))
    push!(ndofs, Int.(readdlm(str*"_ndofs.log")))
    push!(avg_tics, readdlm(str*"_avg_tics.log"))
    push!(tics, readdlm(str*"_tics.log"))
end

labels = [
        L"p"*"-uniform",
        L"h"*"-uniform, "*L"p=1",
        L"h"*"-uniform, "*L"p=2",
        L"h"*"-uniform, "*L"p=3",
        # L"h"*"-uniform, "*L"p=5",
        L"hp"*"-uniform"
]

p = Plots.plot(yaxis=:log10, xaxis=:log10,
    xticks=[1e0,1e1,1e2,1e3,1e4,1e5,1e6],
    yticks=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2],
    xlim=[7e0, 4e5],
    xlabel="Number of dofs for "*L"u_{hp}", ylabel=L"\Vert u - u_{hp} \Vert_{H^1(\Omega)}",
    ylabelfontsize=15,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=8,
    legend=:bottomleft)
for (i, label) in zip(1:length(ndofs), labels)
    p = Plots.plot!(ndofs[i], h1s[i], marker=markers[i], linewidth=2, label=label, grid=true)
end
display(p)

idx = [[5,10,20], [1,2,3,4,5,6,7], [3,4,5,6], [5,6], [3,4,5]]
i=1;[annotate!(ndofs[i][j]-5*ndofs[i][j]/12, h1s[i][j]-0.1*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=2;[annotate!(ndofs[i][j]+5*ndofs[i][j]/12, h1s[i][j]+0.2*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=3;[annotate!(ndofs[i][j]+5*ndofs[i][j]/12, h1s[i][j]+0.2*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=4;[annotate!(ndofs[i][j]+5*ndofs[i][j]/12, h1s[i][j]+0.2*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=5;[annotate!(ndofs[i][j]+5*ndofs[i][j]/12, h1s[i][j]+0.3*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
display(p)

function plot_convergence_triangle!(x1,x2,y1,sl,color;dim=1,labelpos=1/8)
    dx = x2/x1
    y2 = y1/(dx^(1/dim))^sl

    xt = [x1, x1, x2, x1]
    yt = [y1, y2, y2, y1]
    plot!(xt, yt, label="", lw=2, color=color, fillalpha=0.3, seriestype=:shape)
    annotate!(x1-labelpos*x1, (y2/y1)^(1/dim)*y1, Plots.text( "$sl", 9, :black))
end

plot_convergence_triangle!(1e4, 3e4, 7e-2, 1, :orange, dim=2)
plot_convergence_triangle!(4e4, 8e4, 2e-3, 2, :green, dim=2)
plot_convergence_triangle!(9.5e4, 2e5,3e-4, 3, :magenta, dim=2)
plot_convergence_triangle!(4e4, 7e4,7e-5, 4, :brown, dim=2)


Plots.savefig("gradient-bounds-convergence.pdf")