using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/laboratory/data/TaSK"

D = 128
Lx = 64
Ly = 1
J = 1.0
params = (J=J,)

N = 100
k = [0.0,0.0]

figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];
title = "$(Lx)x$(Ly) Squa, D = $(D), $(params), N = $(N)",
xticks = 0:20:100,
figsize...)

for k in 0:0.1:0.9
    kv = [k,0.0]
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(kv)_$(N).jld2" data
    inds = filter(x -> data["S"][x] > 1e-4, 1:length(data["S"]))
    ω = data["ω"][inds]
    S = sqrt.(data["S"][inds])
    scatter!(ax,ones(length(ω)) * k, ω,color = get(colorschemes[:dense],log.(S),(-4,0)),markersize = 8)
end

Colorbar(fig[1,2], colorrange = (-4,0), colormap = colorschemes[:dense])

xlims!(ax,0,1.0)
ylims!(ax,0,4.5)

resize_to_layout!(fig)

display(fig)



