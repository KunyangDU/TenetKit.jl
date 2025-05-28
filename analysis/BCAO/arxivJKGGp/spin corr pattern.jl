using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/BCAO/arxivJKGGp/data"
figurename = "BCAO/arxivJKGGp/figures"
D = 2^6
Lx = 4
Ly = 4
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=0.0,J3xy=0.0,J3z=0.0)
params = (J1=-0.4,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

nb = vcat(neighbor(Latt),neighbor(Latt;level =2),neighbor(Latt;level =3))
xbonds,ybonds,zbonds = getxyzbonds(Latt)
ssx = [gsdata["SxSx"][pair] for pair in nb]
ssy = [gsdata["SySy"][pair] for pair in nb]
ssz = [gsdata["SzSz"][pair] for pair in nb]

figsize = (width = Lx*sqrt(3)*50,height = Ly*60)
fig = Figure()
axx = Axis(fig[1,1];autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (1:0.5:Ly+0.5,string.(1:2Ly)),
title = "$(Ly)x$(Lx)x2 ZZ-HC-CY, D=$(D)")

axy = Axis(fig[2,1];autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (1:0.5:Ly+0.5,string.(1:2Ly)))
axz = Axis(fig[3,1];autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (1:0.5:Ly+0.5,string.(1:2Ly)))

plotLatt!(axx,Latt,[0,1];site = true,sitelabel = false,bond = false)
plotLatt!(axy,Latt,[0,1];site = true,sitelabel = false,bond = false)
plotLatt!(axz,Latt,[0,1];site = true,sitelabel = false,bond = false)

limits = (-0.1,0.1)

plotbond!(axx,Latt,nb,ssx,[0,1];colorlimit = limits,linewidth = 40,colormap = :bwr)
plotbond!(axy,Latt,nb,ssy,[0,1];colorlimit = limits,linewidth = 40,colormap = :bwr)
plotbond!(axz,Latt,nb,ssz,[0,1];colorlimit = limits,linewidth = 40,colormap = :bwr)

Colorbar(fig[1,2],limits = limits,colormap = :bwr,label = L"\langle S_xS_x\rangle")
Colorbar(fig[2,2],limits = limits,colormap = :bwr,label = L"\langle S_yS_y\rangle")
Colorbar(fig[3,2],limits = limits,colormap = :bwr,label = L"\langle S_zS_z\rangle")

hidexdecorations!(axx,grid = false,ticks = false)
hidexdecorations!(axy,grid = false,ticks = false)

resize_to_layout!(fig)
display(fig)

save("$(figurename)/spin corr pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("$(figurename)/spin corr pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)




