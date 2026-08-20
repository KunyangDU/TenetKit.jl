using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/laboratory/data/TaSK/U1"

D = 64
Lx = 64
Ly = 1
J = 1.0
params = (Jz = 1.0,Jxy = 0.5)

N = 100
k = [1.0,0.0]

figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];
title = "$(Lx)x$(Ly) Squa, D = $(D), $(params), N = $(N), k = $(k)",
# xticks = 0:20:100,
# xscale = log10,
# yscale = log10,
figsize...)

@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(N).jld2" data
ω = data["ω"]
S = sqrt.(data["S"]) * data["d"]

# scatterlines!(ax,ω,S)
lsω = 0:0.01:4
lines!(ax,lsω,cfe(data["a"],data["b"][1:end-1],data["d"],lsω;η = 0.12,K=10))
# xlims!(ax,0,1.0)
# ylims!(ax,0,4.5)

resize_to_layout!(fig)

display(fig)

S