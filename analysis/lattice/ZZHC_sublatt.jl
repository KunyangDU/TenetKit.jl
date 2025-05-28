using FiniteLattices,CairoMakie
include("../analysis/analysis.jl")


Lx = 4
Ly = 4
Latt = ZZHoneyComb(Lx,Ly)  
figsize = (width = 40*Lx*3,height = 40*(Ly+1/2)*sqrt(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (0.5 .+ (1:1:Ly),string.(1:Ly))
)
siteA = filter( x -> iseven(Latt[x][1]), 1:size(Latt))
siteB = filter( x -> isodd(Latt[x][1]), 1:size(Latt))
# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
plotLatt!(ax,Latt,[0,1];site = false,tplevel=1)
plotLatt!(ax,Latt,[0,1];bond = false,site = true,selectedsite=siteA,sitecolor = [:red for _ in 1:length(siteA)])
plotLatt!(ax,Latt,[0,1];bond = false,site = true,selectedsite=siteB,sitecolor = [:blue for _ in 1:length(siteB)])

resize_to_layout!(fig)
display(fig)

save("lattice/figures/ZZHC_$(Lx)x$(Ly)_sublatt.png",fig)
save("lattice/figures/ZZHC_$(Lx)x$(Ly)_sublatt.pdf",fig)

