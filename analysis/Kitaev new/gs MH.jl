using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/ZZHC"

Lx = 6
Ly = 3
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 256
lsHc = 0.1:0.1:1.0
lsSp = zeros(length(lsHc),size(Latt))

for (iHc,Hc) in enumerate(lsHc)
params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)
proj = normalize([Hx,Hy,Hz])
direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]
# edgepoints = vcat(1:Ly + 2, size(Latt)-Ly - 1:size(Latt))
edgepoints = []

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata

for i in 1:size(Latt)
lsSp[iHc,i] = proj' * map(x -> real(obs[(x,)][(i,)]),["Sx","Sy","Sz"])
end
end

lsSc = mean(lsSp,dims = 2)[:]

figsize = (width = 400,height = 200)

fig = Figure()

ax = Axis(fig[1,1]; 
# autolimitaspect = true,
figsize...,
# yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
# xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1)),
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
# xgridvisible = false,
# ygridvisible =false,
)

scatterlines!(ax,lsHc,lsSc)

resize_to_layout!(fig)
display(fig)
# save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
# save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)

