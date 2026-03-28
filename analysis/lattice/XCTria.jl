using FiniteLattices,CairoMakie

include("../analysis/analysis.jl")

Lx = 9
Ly = 8
Latt = XCTria(Lx,Ly)
# pairsx,pairsy = _ShastrySutherPairs(Latt)
figsize = (width = 60*Lx,height = 60*(Ly)*sqrt(3)/2)
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (0.5 .+ (1:Lx),string.(1:Lx)),yticks = (sqrt(3)/2*(1:Ly),string.(1:Ly)))

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
plotLatt!(ax,Latt,[0,sqrt(3)/2];site = true,tplevel=1)

resize_to_layout!(fig)
display(fig)

save("lattice/figures/XCTria_$(Lx)x$(Ly).png",fig)
save("lattice/figures/XCTria_$(Lx)x$(Ly).pdf",fig)





