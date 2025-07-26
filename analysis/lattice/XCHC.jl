using FiniteLattices,CairoMakie
include("../analysis/analysis.jl")

Lx = 6
Ly = 6

Latt = XCHoneyComb(Lx,Ly)
 
figsize = (width = 80*(Lx+1),height = 80*(Ly)*sqrt(3)/2)
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1))
)

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
plotLatt!(ax,Latt,[0,sqrt(3)/2];site = true,tplevel=1)

resize_to_layout!(fig)
display(fig)

save("lattice/figures/XCHC_$(Lx)x$(Ly).png",fig)
save("lattice/figures/XCHC_$(Lx)x$(Ly).pdf",fig)

