using Plots, LaTeXStrings, DelimitedFiles

ndofs, h1s, avg_tics = [], [], []

path = "output/1d_high_freq/"
strs = path .* ["hik",
                "pg_h_adaptive_p_uniform_refine_p_1"]

for str in strs
    push!(h1s, readdlm(str*"_h1s.log"))
    push!(ndofs, Int.(readdlm(str*"_ndofs.log")))
    push!(avg_tics, readdlm(str*"_avg_tics.log"))
end

labels = ["(HIK) "*L"h"*"-uniform, "*L"p=1",
         L"h"*"-adaptive (VI), "*L"p"*"-uniform"]

p = Plots.plot(xaxis=:log10, yaxis=:log10, yticks=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
for (i, label) in zip(1:length(ndofs), labels)
    p = Plots.plot!(ndofs[i], h1s[i], marker=:xcross, linewidth=2, label=label)
end
display(p)