using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Heisenberg/data/triangle"

D = 2^7
Lx = 6
Ly = 6
params = (Jz=1,Jxy=0.5,h=0)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

nb = neighbor(Latt)
# ssx = [gsdata["SxSx"][pair] for pair in nb]
# ssy = [gsdata["SySy"][pair] for pair in nb]
ssxy = [(gsdata["S+S-"][pair]+gsdata["S-S+"][pair])/2 for pair in nb]
ssz = [gsdata["SzSz"][pair] for pair in nb]

figsize = (width = 60*Lx,height = 60*(Ly)*sqrt(3)/2)
fig = Figure()

axxy = Axis(fig[1,1];autolimitaspect = true,figsize...,
xticks = (0.5 .+ (1:Lx),string.(1:Lx)),yticks = (sqrt(3)/2*(1:Ly),string.(1:Ly)))

axz = Axis(fig[2,1];autolimitaspect = true,figsize...,
xticks = (0.5 .+ (1:Lx),string.(1:Lx)),yticks = (sqrt(3)/2*(1:Ly),string.(1:Ly)))


plotLatt!(axxy,Latt,[0,sqrt(3)/2];site = true,sitelabel = false,bond = false)
plotLatt!(axz,Latt,[0,sqrt(3)/2];site = true,sitelabel = false,bond = false)

# limits = (-0.1,0.1)
limits = (-1/4,1/4)

plotbond!(axxy,Latt,nb,ssxy,[0,sqrt(3)/2];colorlimit = limits,linewidth = 40,colormap = :bwr)
plotbond!(axz,Latt,nb,ssz,[0,sqrt(3)/2];colorlimit = limits,linewidth = 40,colormap = :bwr)

Colorbar(fig[1,2],limits = limits,colormap = :bwr,label = L"\langle S_xS_x + S_yS_y\rangle")
Colorbar(fig[2,2],limits = limits,colormap = :bwr,label = L"\langle S_zS_z\rangle")

hidexdecorations!(axy,grid = false,ticks = false)

resize_to_layout!(fig)
display(fig)

save("triangle/figures/spin corr pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("triangle/figures/spin corr pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)




