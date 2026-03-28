using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")



Lx = 3
Ly = 2
Latt = PCHoneyComb(Lx,Ly)  
figsize = (width = 40*Lx*3,height = 40*(Ly+1/2)*sqrt(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
# xticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
# yticks = (0.5 .+ (1:1:Ly),string.(1:Ly))
)

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
flux_Latt = PCTria(Lx-1,Ly)
sites = map(x -> collect(coordinate(flux_Latt,x)) + [1,sqrt(3)/3], 1:size(flux_Latt))
polyHexagon!(ax,sites,get(colorschemes[:Reds], randn(length(sites)),(0,1));scale = 1/sqrt(3))
plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1)
plotLatt!(ax,flux_Latt,[1/2,sqrt(3)/2];site = true,tplevel=1,total_shift = [1,sqrt(3)/3])

edge_shift = [1/2,sqrt(3)/2]
d = 1/sqrt(3)
flux_shift = [1,sqrt(3)/3]
direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]]
fluxsites,fluxdirections,direction = getPBCflux(Latt,flux_Latt,direction;d = d,edge_shift = edge_shift,flux_shift = flux_shift)
plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1)
bonds = getxyzbonds(Latt;shift = [1/2,sqrt(3)/2], direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[1/2,sqrt(3)/2];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[1/2,sqrt(3)/2];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[1/2,sqrt(3)/2];colormap = :Blues)

resize_to_layout!(fig)
display(fig)
# fluxsites[end-10],fluxdirections[end-10]
# fluxsites
fluxdirections

abs(norm(coordinate(Latt,4) .+ [Lx-1,Ly] .* edge_shift .- coordinate(flux_Latt,2) .- flux_shift) - d)