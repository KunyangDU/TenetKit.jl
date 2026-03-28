using FiniteLattices,CairoMakie,LinearAlgebra,ColorSchemes
include("../analysis/analysis.jl")

function getxyzbonds(Latt::AbstractLattice;
    shift = [0,1],
    direction = [[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]],tol=1e-8)
    nb = neighbor(Latt)
    _,Ly = get_cellsize(Latt)
    return map(direction) do v
        filter(x -> abs(dot(let
            u = coordinate(Latt,x[1]) .- coordinate(Latt,x[2])
            if abs(u[2]) > 1
                u = u .- sign(u[2])*shift*Ly
            end
            u
        end,v)) < tol ,nb)
    end
end

Lx = 6
Ly = 6

Latt = PCHoneyComb(Lx,Ly)
# Latt = PCTria(Lx,Ly)
figsize = (width = 80*(Lx+1),height = 80*(Ly)*sqrt(3)/2)
fig = Figure()
ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
# yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
# xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1))
)

# plotLatt!(ax,Latt,[0,1];pairs = pairsx)
# plotLatt!(ax,Latt,[0,1];pairs = pairsy)
# inner_points = filter(x -> length(neighbor(Latt,x;level = 2)) == 6, 1:size(Latt))
# outer_points = filter(x -> length(neighbor(Latt,x;level = 2)) != 6, 1:size(Latt))
# lop = filter(x -> x < size(Latt)/2,outer_points)
# rop = filter(x -> x > size(Latt)/2,outer_points)
plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1)
bonds = getxyzbonds(Latt;shift = [0,1], direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])
plotbond!(ax,Latt,bonds[1],ones(length(bonds[1]))/2,[1/2,sqrt(3)/2];colormap = :Reds)
plotbond!(ax,Latt,bonds[2],ones(length(bonds[2]))/2,[1/2,sqrt(3)/2];colormap = :Greens)
plotbond!(ax,Latt,bonds[3],ones(length(bonds[3]))/2,[1/2,sqrt(3)/2];colormap = :Blues)

# plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1, bond = false,selectedsite = inner_points, sitecolor = [:red for _ in eachindex(inner_points)])
# plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1, bond = false,selectedsite = lop, sitecolor = [:blue for _ in eachindex(lop)])
# plotLatt!(ax,Latt,[1/2,sqrt(3)/2];site = true,tplevel=1, bond = false,selectedsite = rop, sitecolor = [:green for _ in eachindex(rop)])

resize_to_layout!(fig)
display(fig)

save("lattice/figures/PCHC_bond_$(Lx)x$(Ly).png",fig)
save("lattice/figures/PCHC_bond_$(Lx)x$(Ly).pdf",fig)


