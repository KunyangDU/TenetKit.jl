using FiniteLattices,CairoMakie
include("../analysis/analysis.jl")

function ZZHoneyComb(L::Int64,W::Int64)
    shift = ((-1/2sqrt(3),1/2),(0.0,0.0),(1/sqrt(3),0.0),(sqrt(3)/2,1/2))
    return CompositeLattice([YCRect(L,W,(sqrt(3),1.0)) for _ in 1:4]..., shift) |> Snake!    
end




Lx = 8
Ly = 4
shift = ((0.0,0.0),(sqrt(3)/6,1/2))
Latt = CompositeLattice([YCTria(Lx,Ly) for _ in 1:2]..., shift) |> Snake!    
figsize = (width = 80*Lx*sqrt(3)/2,height = 80*(Ly+1))
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

