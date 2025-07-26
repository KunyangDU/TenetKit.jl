using FiniteLattices,CairoMakie
include("../analysis/analysis.jl")

Lx = 3
Ly = 3
Latt = ACHoneyComb(Lx,Ly)  
figsize = (width = 40*(Lx+1/2)*sqrt(3),height = 40*(Ly)*(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
yticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
xticks = (0.5 .+ 0.5(1:1:2Lx),string.(1:2Lx))
)

direction = [[1,0],[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2]]
bonds = getxyzbonds(Latt;direction=direction)

plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,sqrt(3)];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,sqrt(3)];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,sqrt(3)];colormap = :Blues)
plotLatt!(ax,Latt,[0,sqrt(3)];site = true,tplevel=1,bond = false)

resize_to_layout!(fig)
display(fig)

save("lattice/figures/ACHC_$(Lx)x$(Ly)_bond.png",fig)
save("lattice/figures/ACHC_$(Lx)x$(Ly)_bond.pdf",fig)


