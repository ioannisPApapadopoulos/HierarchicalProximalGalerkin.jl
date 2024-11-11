using DelimitedFiles
using Plots, LaTeXStrings

path = "output/bessel_obstacle/"

ndofs_p_fem, ndofs_h_fem_p_2, ndofs_h_fem_p_3, ndofs_hik = readdlm(path*"bessel_ndofs_p_fem.log"), 
    readdlm(path*"bessel_ndofs_h_fem_p_2.log"), readdlm(path*"bessel_ndofs_h_fem_p_3.log"), readdlm(path*"bessel_ndofs_hik.log")

h1s_p_fem, h1s_h_fem_p_2, h1s_h_fem_p_3, h1s_hik = readdlm(path*"bessel_h1s_p_fem.log"), 
    readdlm(path*"bessel_h1s_h_fem_p_2.log"), readdlm(path*"bessel_h1s_h_fem_p_3.log"), readdlm(path*"bessel_h1s_hik.log")

avg_tics_p_fem, avg_tics_h_fem_p_2, avg_tics_h_fem_p_3, avg_tics_hik = readdlm(path*"bessel_avg_tics_p_fem.log"), 
    readdlm(path*"bessel_avg_tics_h_fem_p_2.log"), readdlm(path*"bessel_avg_tics_h_fem_p_3.log"), readdlm(path*"bessel_avg_tics_hik.log")



ns = [10;20;40]
ps = 2:12
# Plots.plot(ndofs_p_fem[1:end-1], [h1s_p_fem 2^(2.4) ./ ps.^(2.4) * h1s_p_fem[1]], yaxis=:log10, xaxis=:log10, linestyle=[:solid :dash],  marker=[:dtriangle :none])
# Plots.plot!(ndofs_h_fem_p_3[1:end-1], [h1s_h_fem_p_3 h1s_h_fem_p_3[2]*ns[2].^(2) ./ ns.^(2)], yaxis=:log10, xaxis=:log10, linestyle=[:solid :dash],  marker=[:dtriangle :none])

p = Plots.plot(ndofs_p_fem[1:end-2], [h1s_p_fem[1:end-1]], 
    yaxis=:log10, xaxis=:log10,  
    marker=:dtriangle,
    label="(S1)",
    ylabel=L"\Vert u - u_{hp} \Vert_{H^1(\Omega)}", xlabel="Degrees of freedom",
    ylabelfontsize=15,xlabelfontsize=15, xtickfontsize=10, ytickfontsize=10, 
    legendfontsize=10,
    yticks=[1e-2,10^(-1.5), 1e-1,10^(-0.5),1e0,10^(0.5)],
    # ylim=[10^(-1.5),5e0]
)
Plots.plot!(ndofs_h_fem_p_2[1:end-1], [h1s_h_fem_p_2], label=L"(S2), $p=2$", marker=:square)
Plots.plot!(ndofs_h_fem_p_3[1:end-1], [h1s_h_fem_p_3], label=L"(S2), $p=3$", marker=:dot)
Plots.plot!(ndofs_hik[1:end-1], [h1s_hik], label="(S3)",  marker=:diamond)

[annotate!(ndofs_p_fem[i], h1s_p_fem[i]-0.1*h1s_p_fem[i], Plots.text( "$(round.(avg_tics_p_fem[i], digits=3))", 9, theme_palette(:auto)[1])) for i in 1:10]
[annotate!(ndofs_h_fem_p_2[i], h1s_h_fem_p_2[i]+0.12*h1s_h_fem_p_2[i], Plots.text( "$(round.(avg_tics_h_fem_p_2[i], digits=3))", 9, theme_palette(:auto)[2])) for i in 1:4]
[annotate!(ndofs_h_fem_p_3[i], h1s_h_fem_p_3[i]+0.12*h1s_h_fem_p_3[i], Plots.text( "$(round.(avg_tics_h_fem_p_3[i], digits=3))", 9, theme_palette(:auto)[3])) for i in 1:3]
[annotate!(ndofs_hik[i], h1s_hik[i]-0.1*(-1)^i*h1s_hik[i], Plots.text( "$(round.(avg_tics_hik[i], digits=3))", 9, theme_palette(:auto)[4])) for i in 1:4]
display(p)

Plots.plot!(ndofs_p_fem[3:5], [2 ./ ps[3:5].^(2) * h1s_p_fem[1]],linestyle=:dot, label=L"$O(p^{-2}), O(h^2)$")
Plots.plot!(ndofs_h_fem_p_3[2:3], 25 ./ [20;40].^(1.5) * h1s_h_fem_p_3[1],linestyle=:dash, label=L"$O(h^{3/2})$")
Plots.plot!(ndofs_hik[2:3], 15 ./ [20;40] * h1s_hik[1],linestyle=:dashdot, label=L"$O(h)$", legend=:bottomleft)


Plots.savefig("oscillatory_obstacle_convergence.pdf")