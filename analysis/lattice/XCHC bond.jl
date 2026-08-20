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

function getxyzbonds(Latt::AbstractLattice,direction::Vector,tol::Float64=1e-8)
    nb = neighbor(Latt)
    _,Ly = get_cellsize(Latt)
    return map(direction) do v
        filter(x -> abs(dot(relaVec(Latt,x...),v)) < tol ,nb)
    end
end

Lx = 6
Ly = 2

Latt = XCHoneyComb(Lx,Ly)

ut = vcat(2:2:2Ly,reverse(1:2:2Ly-1),reverse(2Ly+2:2:4Ly),2Ly+1:2:4Ly-1)
pinds = invperm(vcat([(i-1)*4Ly .+ ut for i in 1:Lx]...))
# permute!(Latt,pinds)
 
figsize = (width = 80*(Lx+1),height = 80*(Ly)*sqrt(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1))
)

bonds = getxyzbonds(Latt,[[1,0],[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2]])
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2;colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2;colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2;colormap = :Blues)
plotLatt!(ax,Latt;site = true,tplevel=1,bond = false)
# # plotLatt!(ax,Latt,[0,sqrt(3)/2];site = true,tplevel=1,bond = false,selectedsite = x_stripe_points,sitecolor = [:red for _ in eachindex(x_stripe_points)])
# plotLatt!(ax,Latt,[0,sqrt(3)/2];site = true,tplevel=1,bond = false,selectedsite = 4Ly:size(Latt)-4Ly,sitecolor = [:red for _ in eachindex(4Ly:size(Latt)-4Ly)])


resize_to_layout!(fig)
display(fig)

save("lattice/figures/XCHC_$(Lx)x$(Ly).png",fig)
save("lattice/figures/XCHC_$(Lx)x$(Ly).pdf",fig)



