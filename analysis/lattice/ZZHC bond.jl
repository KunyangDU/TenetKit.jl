using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")
function getxyzbonds(Latt::AbstractLattice,direction::Vector,tol::Float64=1e-8)
    nb = neighbor(Latt)
    _,Ly = get_cellsize(Latt)
    return map(direction) do v
        filter(x -> abs(dot(relaVec(Latt,x...),v)) < tol ,nb)
    end
end


Lx = 6
Ly = 3
Latt = ZZHoneyComb(Lx,Ly)  

ut = vcat(2:2:2Ly,reverse(1:2:2Ly-1),reverse(2Ly+2:2:4Ly),2Ly+1:2:4Ly-1)
pinds = invperm(vcat([(i-1)*4Ly .+ ut for i in 1:Lx]...))
permute!(Latt,pinds)

figsize = (width = 40*Lx*3,height = 40*(Ly+1/2)*sqrt(3))
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (0.5 .+ (1:1:Ly),string.(1:Ly))
)
# direction=[[0,1],[sqrt(3)/2,-1/2],[sqrt(3)/2,1/2]]
# direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]
# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
ldirection=[[0,1],[sqrt(3)/2,-1/2],[sqrt(3)/2,1/2]]
rdirection=[[0,1],[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2]]
lsites = collect(1:div(size(Latt),2))
rsites = collect(div(size(Latt),2)+1:size(Latt))
lbonds = getxyzbonds(Latt, ldirection)
rbonds = getxyzbonds(Latt, rdirection)
bonds = map(1:3) do i
    vcat(filter(y -> length(intersect(y,lsites)) ≥ 1,lbonds[i]),filter(y -> length(intersect(y,rsites)) == 2,rbonds[i]))
end

plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2;colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2;colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2;colormap = :Blues)
plotLatt!(ax,Latt;site = true,tplevel=1, bond = false)

unitcol = collect(1:2Ly)
uppts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[2,3,6])...)
# dnpts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[3,10])...)
# zeropts = vcat(map(x -> (x-1)*2Ly .+ unitcol,[4,6,7,9])...)

# plotLatt!(ax,Latt;bond = false, site = true, selectedsite = uppts, sitelabel = false, sitecolor = [:red for _ in eachindex(uppts)])
# plotLatt!(ax,Latt,[0,1];bond = false, site = true, selectedsite = dnpts, sitelabel = false, sitecolor = [:blue for _ in eachindex(dnpts)])
# plotLatt!(ax,Latt,[0,1];bond = false, site = true, selectedsite = zeropts, sitelabel = false, sitecolor = [:grey for _ in eachindex(zeropts)])


resize_to_layout!(fig)
display(fig)

save("lattice/figures/ZZHC_$(Lx)x$(Ly).png",fig)
save("lattice/figures/ZZHC_$(Lx)x$(Ly).pdf",fig)


# uppts

