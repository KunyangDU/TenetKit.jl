using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")



Lx = 2
Ly = 2
Latt = ZZHoneyComb(Lx,Ly)  
figsize = (width = 40*Lx*3,height = 40*(Ly+1/2)*sqrt(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (0.5 .+ (1:1:Ly),string.(1:Ly))
)

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
flux_Latt = YCTria(2Lx-1,Ly)
sites = map(x -> collect(coordinate(flux_Latt,x)) + [2*sqrt(3)/3,0], 1:size(flux_Latt))
polyHexagon!(ax,sites,get(colorschemes[:Reds], randn(length(sites)),(0,1));scale = 1/sqrt(3),rot_mat = [0 -1;1 0])
plotLatt!(ax,Latt,[0,1];site = true,tplevel=1)
plotLatt!(ax,flux_Latt,[0,1];site = true,tplevel=1,total_shift = [2*sqrt(3)/3,0])

edge_shift = [0,1]
d = 1/sqrt(3)
flux_shift = [2*sqrt(3)/3,0]
direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]
fluxsites,fluxdirections,direction = getPBCflux(Latt,flux_Latt,[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]];d = 1/sqrt(3),edge_shift = [0,1],flux_shift = [2*sqrt(3)/3,0])

bonds = getxyzbonds(Latt;shift = [0,1], direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]])
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,1];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,1];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,1];colormap = :Blues)

resize_to_layout!(fig)
display(fig)
# fluxsites[end-10],fluxdirections[end-10]
# fluxsites
fluxdirections