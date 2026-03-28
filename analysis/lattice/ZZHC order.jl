using FiniteLattices,CairoMakie
include("../analysis/analysis.jl")


Lx = 6
Ly = 4
Latt = ZZHoneyComb(Lx,Ly)  
figsize = (width = 40*Lx*3,height = 40*(Ly+1/2)*sqrt(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (0.5 .+ (1:1:Ly),string.(1:Ly))
)

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
plotLatt!(ax,Latt,[0,1];site = false,tplevel=1)
bonds = getxyzbonds(Latt;shift = [0,1], direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]])
# plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,1];colormap = :Reds)
# plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,1];colormap = :Greens)
# plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,1];colormap = :Blues)

# unitcol = collect(1:8)
# uppts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[1,2,5,8,11,12])...)
# dnpts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[3,10])...)
# zeropts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[4,6,7,9])...)

# unitcol = collect(1:Ly)
# uppts = vcat(map(x -> (x-1)*Ly .+ unitcol,[1,2,5,8,11,12])...)
# dnpts = vcat(map(x -> (x-1)*Ly .+ unitcol,[3,10])...)
# zeropts = vcat(map(x -> (x-1)*Ly .+ unitcol,[4,6,7,9])...)

# unitcol = collect(1:2Ly)
# uppts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[1,2,4,5,7,8,10,11])...)
# dnpts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[3,6,9,12])...)
# zeropts = []

unitcol = collect(1:2Ly)
uppts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[1,2,4,5,8,9,11,12])...)
dnpts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[3,6,7,10])...)
zeropts = []

# unitcol = collect(1:2Ly)
# uppts = vcat(map(x -> (x-1)*2Ly .+ unitcol, [2i-1 for i in 1:Lx])...)
# dnpts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[2i for i in 1:Lx])...)
# zeropts = []

plotLatt!(ax,Latt,[0,1];bond = false, site = true, selectedsite = uppts, sitelabel = false, sitecolor = [:red for _ in eachindex(uppts)])
plotLatt!(ax,Latt,[0,1];bond = false, site = true, selectedsite = dnpts, sitelabel = false, sitecolor = [:blue for _ in eachindex(dnpts)])
plotLatt!(ax,Latt,[0,1];bond = false, site = true, selectedsite = zeropts, sitelabel = false, sitecolor = [:grey for _ in eachindex(zeropts)])


resize_to_layout!(fig)
display(fig)

save("lattice/figures/ZZHC_NNBO_$(Lx)x$(Ly).png",fig)
save("lattice/figures/ZZHC_NNBO_$(Lx)x$(Ly).pdf",fig)


# uppts