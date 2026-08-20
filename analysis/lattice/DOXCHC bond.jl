using FiniteLattices,CairoMakie,ColorSchemes
include("../analysis/analysis.jl")
function getxyzbonds(Latt::AbstractLattice,direction::Vector,tol::Float64=1e-8)
    nb = neighbor(Latt)
    _,Ly = get_cellsize(Latt)
    return map(direction) do v
        filter(x -> abs(dot(relaVec(Latt,x...),v)) < tol ,nb)
    end
end



Lx = 8
Ly = 1
Latt = DiamondOpenXCHoneyComb(Lx,Ly)  
figsize = (width = 80*Lx,height = 160*Ly*sqrt(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
# xticks = (5.5 .+ 1.0 .* (1:Lx),string.(1:Lx)),
yticks = (sqrt(3)/6 + sqrt(3) .+ sqrt(3) * (0:Ly-1),string.(1:Ly))
)
direction = [[1,0],[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2]]

bonds = getxyzbonds(Latt,direction)
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2;colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2;colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2;colormap = :Blues)
plotLatt!(ax,Latt;site = true,tplevel=1, bond = false)


# unitcol = collect(1:8)
# uppts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[1,2,5,8,11,12])...)
# dnpts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[3,10])...)
# zeropts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[4,6,7,9])...)

# plotLatt!(ax,Latt;bond = false, site = true, selectedsite = uppts, sitelabel = false, sitecolor = [:red for _ in eachindex(uppts)])
# plotLatt!(ax,Latt;bond = false, site = true, selectedsite = dnpts, sitelabel = false, sitecolor = [:blue for _ in eachindex(dnpts)])
# plotLatt!(ax,Latt;bond = false, site = true, selectedsite = zeropts, sitelabel = false, sitecolor = [:grey for _ in eachindex(zeropts)])


resize_to_layout!(fig)
display(fig)

save("lattice/figures/ZZHC_$(Lx)x$(Ly).png",fig)
save("lattice/figures/ZZHC_$(Lx)x$(Ly).pdf",fig)

cornerpoints = vcat(1:6Ly,size(Latt)-6Ly+1:size(Latt))
xedgepoints = filter(x -> length(neighbor(Latt,x))==2 && x ∈ cornerpoints,1:size(Latt))
yedgepoints = filter(x -> length(neighbor(Latt,x))==2 && x ∉ cornerpoints,1:size(Latt))
coordinate(Latt,1)
# uppts