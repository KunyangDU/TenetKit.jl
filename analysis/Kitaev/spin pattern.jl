using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data"

D = 2^7
Lx = 4
Ly = 4
params = (Jx=0.5,Jy=0.5,Jz=1.4)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/sweep/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/sweep/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
@load "$(dataname)/sweep/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

xbonds,ybonds,zbonds = getxyzbonds(Latt)
selectedpoints = 17:48
ssx = [gsdata["SxSx"][pair] for pair in xbonds]
ssy = [gsdata["SySy"][pair] for pair in ybonds]
ssz = [gsdata["SzSz"][pair] for pair in zbonds]

figsize = (width = Lx*sqrt(3)*60,height = Ly*60)
fig = Figure()
ax = Axis(fig[1,1:3];autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (1:0.5:Ly+0.5,string.(1:2Ly)),
title = "$(Lx)x$(Ly) Kitaev model, D=$(D), $(params)")

plotLatt!(ax,Latt,[0,1];site = true,sitelabel = false)
plotbond!(ax,Latt,xbonds,-4ssx,[0,1];colorlimit = (0.,1.),linewidth = 8,colormap = :Reds)
plotbond!(ax,Latt,ybonds,-4ssy,[0,1];colorlimit = (0.,1.),linewidth = 8,colormap = :Blues)
plotbond!(ax,Latt,zbonds,-4ssz,[0,1];colorlimit = (0.,1.),linewidth = 8,colormap = :Greens)

Colorbar(fig[2,1],limits = (0,1),width = Lx*sqrt(3)*18,height = 10,vertical=false,flipaxis = false,colormap = :Reds,label = L"\langle S_xS_x\rangle")
Colorbar(fig[2,2],limits = (0,1),width = Lx*sqrt(3)*18,height = 10,vertical=false,flipaxis = false,colormap = :Blues,label = L"\langle S_yS_y\rangle")
Colorbar(fig[2,3],limits = (0,1),width = Lx*sqrt(3)*18,height = 10,vertical=false,flipaxis = false,colormap = :Greens,label = L"\langle S_zS_z\rangle")

resize_to_layout!(fig)
display(fig)

save("Kitaev/figures/spin pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Kitaev/figures/spin pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

