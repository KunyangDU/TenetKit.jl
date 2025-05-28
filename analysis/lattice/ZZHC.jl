using FiniteLattices,CairoMakie
include("../analysis/analysis.jl")


Lx = 5
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
plotLatt!(ax,Latt,[0,1];site = true,tplevel=1)

resize_to_layout!(fig)
display(fig)

