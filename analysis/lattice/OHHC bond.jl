using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")



L = 2
Latt = OHHoneyComb(L)

fig = Figure()
figsize = (height = 200, width = 200)

ax = Axis(fig[1,1];autolimitaspect = true,figsize...)
bonds = getxyzbonds(Latt;shift = [0,0], direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])

area = Dict(
    2 => Dict(
        "A" => [2,3,4,10,11,18],
        "B" => [1,5,6,7,14,15],
        "C" => [19,20,21,22,23,24],
        "D" => [8,9,12,13,16,17]
    ),
    3 => Dict(
        "A" => [2,3,4,11,12,13,14,23,24,25,36,37],
        "B" => [1,5,6,7,8,17,18,19,20,30,31,42],
        "C" => [43,44,45,46,47,48,49,50,51,52,53,54],
        "D" => [9,10,15,16,21,22,26,27,28,29,32,33,34,35,38,39,40,41]
    )
)
# polyHexagon!(ax,sites,get(colorschemes[:Reds], randn(length(sites)),(0,1)))
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,0];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,0];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,0];colormap = :Blues)
plotLatt!(ax,Latt,[0,0];site = true,bond = false)

resize_to_layout!(fig)
display(fig)

save("lattice/figures/OHHC_bond_$(L).png",fig)
save("lattice/figures/OHHC_bond_$(L).pdf",fig)
