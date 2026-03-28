using FiniteLattices,CairoMakie
include("../analysis/analysis.jl")




Lx = 6
Ly = 6

Latt = PCHoneyComb(Lx,Ly)
# Latt = PCTria(Lx,Ly)
figsize = (width = 80*(Lx+1),height = 80*(Ly)*sqrt(3)/2)
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
# yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
# xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1))
)

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
inner_points = filter(x -> length(neighbor(Latt,x;level = 2)) == 6, 1:size(Latt))
outer_points = filter(x -> length(neighbor(Latt,x;level = 2)) != 6, 1:size(Latt))
lop = filter(x -> x < size(Latt)/2,outer_points)
rop = filter(x -> x > size(Latt)/2,outer_points)
plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1)
# plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1, bond = false,selectedsite = inner_points, sitecolor = [:red for _ in eachindex(inner_points)])
# plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1, bond = false,selectedsite = lop, sitecolor = [:blue for _ in eachindex(lop)])
# plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1, bond = false,selectedsite = rop, sitecolor = [:green for _ in eachindex(rop)])

resize_to_layout!(fig)
display(fig)

save("lattice/figures/PCHC_$(Lx)x$(Ly).png",fig)
save("lattice/figures/PCHC_$(Lx)x$(Ly).pdf",fig)


