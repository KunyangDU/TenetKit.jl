using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/laboratory/data/TaSK"
figurename = "Heisenberg/TaSK/figures"

D = 400
Lx = 64
Ly = 1
J = 1.0
params = (J=J, )

N = 100
k = [1.0,0.0]

figsize = (width = 500,height = 300)

fig = Figure()
ax = Axis(fig[1,1];
title = "$(Lx)x$(Ly) Squa, D = $(D), $(params), k = $(k)",
xlabel = L"N",ylabel = L"\omega",
xticks = 0:20:100,
figsize...)

lsS = zeros(N)

for i in 1:N
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(i).jld2" data
    inds = filter(x -> data["S"][x] > 1e-5, 1:length(data["S"]))
    ω = data["ω"][inds]
    S = data["S"][inds] * data["d"] ^ 2
    scatter!(ax,ones(length(ω)) * i, ω,color = get(colorschemes[:dense],log.(S),(-4,0)))
    lsS[i] = data["S"][1]
end

Colorbar(fig[1,2], colorrange = (-4,0), colormap = colorschemes[:dense],label = L"\ln\ S_\alpha")

xlims!(ax,0,N)
ylims!(ax,0,4)

resize_to_layout!(fig)

display(fig)

save("$(figurename)/Lanczos_level_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(N).png",fig)
save("$(figurename)/Lanczos_level_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(N).pdf",fig)


