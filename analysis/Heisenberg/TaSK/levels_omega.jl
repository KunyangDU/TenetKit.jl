using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/laboratory/data/TaSK"
figurename = "Heisenberg/TaSK/figures"
D = 64
Lx = 64
Ly = 1
J = 1.0
params = (J=J,)

N = 100
k = [1.0,0.0]

figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];
title = "$(Lx)x$(Ly) Squa, D = $(D), $(params), N = $(N), k = $(k)",
# xticks = 0:20:100,
xscale = log10,
yscale = log10,
figsize...)

@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(N).jld2" data
inds = filter(x -> data["S"][x] > 1e-4, 1:length(data["S"]))
ω = data["ω"][inds]
S = sqrt.(data["S"][inds])
scatterlines!(ax,ω,S)

# xlims!(ax,0,1.0)
# ylims!(ax,0,4.5)

resize_to_layout!(fig)

display(fig)
save("$(figurename)/S_omega_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(N).png",fig)
save("$(figurename)/S_omega_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(N).pdf",fig)



