using Plots, LaTeXStrings, DelimitedFiles

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
    xticks=[1e0,1e1,1e2,1e3,1e4,1e5],
    xlabel="Degrees of freedom", ylabel=L"\Vert u - u_{hp} \Vert_{H^1(\Omega)}",
    ylabelfontsize=15,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=8)

for (i, label) in zip(1:length(ndofs), labels)
    p = Plots.plot!(ndofs[i], h1s[i], marker=markers[i], linewidth=2, label=label, grid=true)
    # [annotate!(ndofs[i][j], h1s[i][j]-0.4*h1s[i][j], Plots.text( "$(round.(avg_tics[i][j], digits=3))", 9, theme_palette(:auto)[i])) for j in 2:2:length(ndofs[i])]
end
display(p)
pw=10;i=7;j=3;
Plots.plot!(ndofs[i][j:end], h1s[i][j] * (ndofs[i][j] ./ ndofs[i][j:end]).^(pw),linestyle=:dot, marker=:dot,linewidth=2)
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