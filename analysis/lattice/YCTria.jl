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

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
plotLatt!(ax,Latt;site = true,tplevel=1)

resize_to_layout!(fig)
display(fig)

save("lattice/figures/YCTria_$(Lx)x$(Ly).png",fig)
save("lattice/figures/YCTria_$(Lx)x$(Ly).pdf",fig)
