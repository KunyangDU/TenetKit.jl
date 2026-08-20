using FiniteLattices,CairoMakie

include("../analysis/analysis.jl")

Lx = 12
Ly = 4
Latt = YCTria(Lx,Ly)
# pairsx,pairsy = _ShastrySutherPairs(Latt)
figsize = (width = 80*Lx*sqrt(3)/2,height = 80*(Ly+1))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (1:0.5:Lx+0.5,string.(1:2Lx)),yticks = (1:0.5:Ly+0.5,string.(1:2Ly)))

directions = [[1,0],[-1/2,sqrt(3)/2],[-1/2,-sqrt(3)/2]]

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
plotLatt!(ax,Latt;site = true,tplevel=1,bond = false)
bonds = getxyzbonds(Latt,directions)

plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2;colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2;colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2;colormap = :Blues)

resize_to_layout!(fig)
display(fig)

save("lattice/figures/YCTria_$(Lx)x$(Ly).png",fig)
save("lattice/figures/YCTria_$(Lx)x$(Ly).pdf",fig)
