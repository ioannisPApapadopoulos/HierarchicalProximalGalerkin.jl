using Plots, LaTeXStrings, DelimitedFiles

"""
Section 7.1

Helper script for plotting convergence of oscillatory right-hand side.
"""

path = "output/1d_high_freq/"
markers = [:circle, :rect,  :diamond, :hexagon, :star5, :cross,:utriangle,  :xcross, 
    :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, 
    :star7, :star8, :vline, :hline, :+, :x]

ndofs, h1s, avg_tics = [], [], []
strs = path .* ["hik",
                "hik_adaptive",
                # "pg_h_adaptive_refine_p_3",
                # "pg_h_adaptive_p_uniform_refine_p_1",
                "pg_h_uniform_refine_p_1",
                "pg_h_uniform_refine_p_3",
                "pg_p_uniform_refine",
                "pg_h_adaptive_p_uniform_refine_p_3",
                "pg_hp_refine"
]

for str in strs[1:end]
    push!(h1s, readdlm(str*"_h1s.log"))
    push!(ndofs, Int.(readdlm(str*"_ndofs.log")))
    push!(avg_tics, readdlm(str*"_avg_tics.log"))
end

labels = ["(PDAS) "*L"h"*"-uniform, "*L"p=1",
            "(PDAS) "*L"h"*"-adaptive, "*L"p=1",
        #  L"h"*"-adaptive, "*L"p=4",
        #  L"h"*"-adaptive, "*L"p"*"-uniform",
        L"h"*"-uniform, "*L"p=2",
        L"h"*"-uniform, "*L"p=4",
        L"p"*"-uniform",
         L"h"*"-adaptive, "*L"p"*"-uniform",
         L"hp"*"-adaptive"
]

p = Plots.plot(yaxis=:log10, xaxis=:log10,
    # xlim=[0,700],
    yticks=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2], ylim=[1e-4,1e2],
    xlim=[7e0, 1e5], 
    xticks=[1e0,1e1,1e2,1e3,1e4,1e5,1e6],
    xlabel="Number of dofs for "*L"u_{hp}", ylabel=L"\Vert u - u_{hp} \Vert_{H^1(\Omega)}",
    ylabelfontsize=15,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=8)
for (i, label) in zip(1:length(ndofs), labels)
    p = Plots.plot!(ndofs[i], h1s[i], marker=markers[i], linewidth=2, label=label, grid=true)
end
idx = [[13], [24], [8], [6], [40], [6,9], []]
i=1; [annotate!(ndofs[i][j]+5*ndofs[i][j]/12, h1s[i][j]+0.3*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j]*1e3, digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=2; [annotate!(ndofs[i][j], h1s[i][j]-0.4*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j]*1e3, digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=3; [annotate!(ndofs[i][j]+5*ndofs[i][j]/12, h1s[i][j], Plots.text( "$(round.(avg_tics[i][j]*1e3, digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=4; [annotate!(ndofs[i][j], h1s[i][j]-0.5*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j]*1e3, digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=5; [annotate!(ndofs[i][j], h1s[i][j]-0.3*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j]*1e3, digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
i=6; [annotate!(ndofs[i][j]-ndofs[i][j]/3, h1s[i][j], Plots.text( "$(round.(avg_tics[i][j]*1e3, digits=2))", 8, theme_palette(:auto)[i])) for j in idx[i]]
display(p)

function plot_convergence_triangle!(x1,x2,y1,sl,color;dim=1,labelpos=1/8)
    dx = x2/x1
    y2 = y1/(dx^(1/dim))^sl

    xt = [x1, x1, x2, x1]
    yt = [y1, y2, y2, y1]
    plot!(xt, yt, label="", lw=2, color=color, fillalpha=0.3, seriestype=:shape)
    annotate!(x1-labelpos*x1, (y2/y1)^(1/2)*y1, Plots.text( "$sl", 9, :black))
end

plot_convergence_triangle!(1e4,2e4,2e-2,1,:blue)
plot_convergence_triangle!(7e2,1.5e3,2e-1,1.5,:red,labelpos=0.2)
plot_convergence_triangle!(1e2,1.5e2,5e-2,5,:green)
plot_convergence_triangle!(9e2,1.1e3,2e-3,10,:magenta,labelpos=0.2)
Plots.savefig("osc-data-convergence.pdf")

#### Comparison of a posteriori estimates

ndofs, h1s, avg_tics = [], [], []
strs = path .* [
                # "pg_h_adaptive_refine_p_3",
                # "pg_h_adaptive_p_uniform_refine_p_1",
                "pg_hp_refine",
                "pg_hp_refine_pg_true"
                ]

for str in strs[1:end]
    push!(h1s, readdlm(str*"_h1s.log"))
    push!(ndofs, Int.(readdlm(str*"_ndofs.log")))
    push!(avg_tics, readdlm(str*"_avg_tics.log"))
end
labels = [
         L"hp"*"-adaptive (VI)",
         L"hp"*"-adaptive (PG)"
        ]

p = Plots.plot(xaxis=:log10, yaxis=:log10, 
    yticks=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2], ylim=[1e-4,1e2],
    xlim=[1e1,1e3],
    xticks=[1e0,1e1,1e2,1e3,1e4,1e5],
    xlabel="Degrees of freedom", ylabel=L"\Vert u - u_{hp} \Vert_{H^1(\Omega)}",
    ylabelfontsize=15,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=10,)

for (i, label) in zip(1:length(ndofs), labels)
    p = Plots.plot!(ndofs[i], h1s[i], marker=markers[i], linewidth=2, label=label)
    # [annotate!(ndofs[i][j], h1s[i][j]-0.4*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=3))", 9, theme_palette(:auto)[i])) for j in 2:2:length(ndofs[i])]
end
display(p)