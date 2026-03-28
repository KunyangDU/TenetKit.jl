using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")

function _x_stripe_points_XCHC(Lx::Int64,Ly::Int64,shift::Int64 = 0)
    if iseven(shift)
        mshift = mod(div(shift,2), div(Ly,2))
        return vcat(1 + mshift, [((2i+1)Ly + div(Ly,2) - mshift*2) .+ (0:1)*(1 + mshift*4) for i in 0:(Lx-2)]..., size(Latt) - div(Ly,2) - mshift*2)
    else
        mshift = mod(div(shift-1,2), div(Ly,2)) + 1
        return vcat([((2i+1)Ly + div(Ly,2) - mshift*2 + 1) .+ (0:1)*(4*mshift - 1) for i in 0:Lx-2]...,size(Latt) - div(Ly,2) - mshift*2 + 1, size(Latt) - div(Ly,2) + mshift)
    end
end

Lx = 4
Ly = 4

Latt = XCHoneyComb(Lx,Ly)
 
figsize = (width = 80*(Lx+1),height = 80*(Ly)*sqrt(3)/2)
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1))
)

x_stripe_points = _x_stripe_points_XCHC(Lx,Ly,3)
@show x_stripe_points
bonds = getxyzbonds(Latt;
shift = [0,sqrt(3)/2],
direction=[[1,0],[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2]])
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[0,sqrt(3)/2];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[0,sqrt(3)/2];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[0,sqrt(3)/2];colormap = :Blues)
# plotLatt!(ax,Latt,[0,sqrt(3)/2];site = true,tplevel=1,bond = false)
# # plotLatt!(ax,Latt,[0,sqrt(3)/2];site = true,tplevel=1,bond = false,selectedsite = x_stripe_points,sitecolor = [:red for _ in eachindex(x_stripe_points)])
# plotLatt!(ax,Latt,[0,sqrt(3)/2];site = true,tplevel=1,bond = false,selectedsite = 4Ly:size(Latt)-4Ly,sitecolor = [:red for _ in eachindex(4Ly:size(Latt)-4Ly)])



resize_to_layout!(fig)
display(fig)

save("lattice/figures/XCHC_$(Lx)x$(Ly).png",fig)
save("lattice/figures/XCHC_$(Lx)x$(Ly).pdf",fig)

div(size(Latt),2) |> x -> [x, x+Ly+1]