using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")

# 从该点出发向其他点的向量求和，检查bond方向，就是对应onsite算符类型
# unique(neighbor), issubset([1,2,3],[1,2,3,4])

L = 3
Latt = OHHoneyComb(L)
direction = [[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]]

flux_Latt = _OHTria(L,((1.0, 0.0),(1/2, sqrt(3)/2));scale = sqrt(3))
fluxsites,fluxdirections,_ = getFlux(Latt,flux_Latt,direction)
@show fluxsites
@show fluxdirections

fig = Figure()
figsize = (height = 300, width = 300)
ax = Axis(fig[1,1];autolimitaspect = true,figsize...)
bonds = getxyzbonds(Latt;shift = [0,0], direction=direction)

sites = map(x -> collect(coordinate(flux_Latt,x)), 1:size(flux_Latt))
polyHexagon!(ax,sites,get(colorschemes[:Reds], randn(length(sites)),(0,1)))
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,0];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,0];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,0];colormap = :Blues)
plotLatt!(ax,Latt,[0,0];site = true, bond = false)


resize_to_layout!(fig)
display(fig)
# color

