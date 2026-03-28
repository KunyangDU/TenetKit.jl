using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")



L = 3
Latt = OHHoneyComb(L)

fig = Figure()
figsize = (height = 300, width = 300)

ax = Axis(fig[1,1];autolimitaspect = true,figsize...)
bonds = getxyzbonds(Latt;shift = [0,0], direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,0];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,0];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,0];colormap = :Blues)
plotLatt!(ax,Latt,[0,0];site = true,bond = false)

resize_to_layout!(fig)
display(fig)

save("lattice/figures/OHHC_bond_$(L).png",fig)
save("lattice/figures/OHHC_bond_$(L).pdf",fig)
