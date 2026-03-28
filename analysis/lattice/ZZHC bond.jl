using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")


Lx = 5
Ly = 6
Latt = ZZHoneyComb(Lx,Ly)  
figsize = (width = 40*Lx*3,height = 40*(Ly+1/2)*sqrt(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (0.5 .+ (1:1:Ly),string.(1:Ly))
)

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
bonds = getxyzbonds(Latt;shift = [0,1], direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]])
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,1];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,1];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,1];colormap = :Blues)
plotLatt!(ax,Latt,[0,1];site = true,tplevel=1, bond = false)

unitcol = collect(1:2Ly)
uppts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[2,3,6])...)
# dnpts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[3,10])...)
# zeropts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[4,6,7,9])...)

plotLatt!(ax,Latt,[0,1];bond = false, site = true, selectedsite = uppts, sitelabel = false, sitecolor = [:red for _ in eachindex(uppts)])
# plotLatt!(ax,Latt,[0,1];bond = false, site = true, selectedsite = dnpts, sitelabel = false, sitecolor = [:blue for _ in eachindex(dnpts)])
# plotLatt!(ax,Latt,[0,1];bond = false, site = true, selectedsite = zeropts, sitelabel = false, sitecolor = [:grey for _ in eachindex(zeropts)])


resize_to_layout!(fig)
display(fig)

save("lattice/figures/ZZHC_$(Lx)x$(Ly).png",fig)
save("lattice/figures/ZZHC_$(Lx)x$(Ly).pdf",fig)


# uppts