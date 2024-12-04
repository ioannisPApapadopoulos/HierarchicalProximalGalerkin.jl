using DelimitedFiles
using Plots, LaTeXStrings

"""
Helper script for plotting convergence of bessel obstacle problem
"""

function plot_convergence_triangle!(x1,x2,y1,sl,color;dim=1,labelpos=1/8)
    dx = x2/x1
    y2 = y1/(dx^(1/dim))^sl

    xt = [x1, x1, x2, x1]
    yt = [y1, y2, y2, y1]
    plot!(xt, yt, label="", lw=2, color=color, fillalpha=0.3, seriestype=:shape)
    annotate!(x1-labelpos*x1, (y2/y1)^(1/dim)*y1, Plots.text( "$sl", 9, :black))
end

path = "output/bessel_obstacle/"

markers = [:circle, :dtriangle, :rect, :star5, :diamond, :hexagon, :cross,:utriangle,  :xcross, 
    :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, 
    :star7, :star8, :vline, :hline, :+, :x]

ndofs, h1s, avg_tics = [], [], []
strs = path .* ["hik_uniform",
                "p_uniform",
                "h_uniform_p_2",
                "h_uniform_p_3",
                "hp_uniform"
]

for str in strs[1:end]
    push!(h1s, readdlm(str*"_h1s.log"))
    push!(ndofs, Int.(readdlm(str*"_ndofs.log")))
    push!(avg_tics, readdlm(str*"_avg_tics.log"))
end

ndofs[2] = ndofs[2][1:end-1]
h1s[2] = h1s[2][1:end-1]

labels = [
        "(PDAS) "*L"h"*"-uniform, "*L"p=1",
        L"p"*"-uniform",
        L"h"*"-uniform, "*L"p=2",
        L"h"*"-uniform, "*L"p=3",
        L"hp"*"-uniform"

]

p = Plots.plot(yaxis=:log10, xaxis=:log10,
    xlim=[7e1,2e7], xticks=[1e1,1e2,1e3,1e4,1e5,1e6,1e7],
    xlabel="Number of dofs for "*L"u_{hp}", ylabel=L"\Vert u - u_{hp} \Vert_{H^1(\Omega)}",
    ylabelfontsize=15,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=8)
for (i, label) in zip(1:length(ndofs), labels)
    p = Plots.plot!(ndofs[i][1:end-1], h1s[i], marker=markers[i], linewidth=2, label=label, grid=true)
end
display(p)

idx = [[4,5,6,7,8,9], [18,23], [3,4,5, 6], [5], [4,5]]
i=1;[annotate!(ndofs[i][j]+5*ndofs[i][j]/12, h1s[i][j]+0.2*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=2;[annotate!(ndofs[i][j]-5*ndofs[i][j]/12, h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=3;[annotate!(ndofs[i][j]+8*ndofs[i][j]/12, h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=4;[annotate!(ndofs[i][j]+0.8*ndofs[i][j], h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=5;[annotate!(ndofs[i][j]-5*ndofs[i][j]/12, h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
display(p)

plot_convergence_triangle!(1e5, 2e5, 7e-2, 1, :blue, dim=2)
plot_convergence_triangle!(1.5e5, 3e5, 6e-3, 3/2, :red, dim=2, labelpos=1/4)

Plots.savefig("oscillatory_obstacle_convergence.pdf")