using FiniteLattices,CairoMakie
include("../analysis/analysis.jl")


Lx = 6
Ly = 4
Latt = YCHoneycomb(Lx,Ly)

figsize = (width = 80*Lx*sqrt(3)/2,height = 80*(Ly+1))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (0.5 .+ (1:1:Ly),string.(1:Ly))
)
triavals = [(-1/2,-sqrt(3)/2),(1,0),(-1/2,sqrt(3)/2)]
direction = [[0,1],[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2]]
direction = [[sqrt(3)/2,-1/2],[0,1],[sqrt(3)/2,1/2]]
bonds = getxyzbonds(Latt;direction=direction)
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,1];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,1];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,1];colormap = :Blues)
plotLatt!(ax,Latt,[0,1];site = true,tplevel=1,bond = false)

resize_to_layout!(fig)
display(fig)

save("lattice/figures/YCHC_$(Lx)x$(Ly).png",fig)
save("lattice/figures/YCHC_$(Lx)x$(Ly).pdf",fig)


