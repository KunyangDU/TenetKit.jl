"""
Shastry-Suther lattice with YC boundary condition.
"""
function YCSS(Lx::Int64,Ly::Int64;path::Function = Snake!)
    Latt = CompositeLattice([YCSqua(Lx,Ly) for _ in 1:4]...,((0.,0.),(0.,0.5),(0.5,0.),(0.5,0.5))) |> path 
    return Latt
end
"""
get Shastry-Suther NNN bond (t′).
"""
function _ShastrySutherPairs(Latt::CompositeLattice{2,4})
    _,Ly = get_cellsize(Latt)
    pairs = interpair(Latt;level=2)
    points1 = filter!(x -> Latt[x][1] ∈ [1,4], collect(1:size(Latt)))
    points2 = filter!(x -> Latt[x][1] ∈ [2,3], collect(1:size(Latt)))
    pairx = filter(x -> x[1] ∈ points1 && x[2] ∈ points1 && isequal(abs.(Latt[x[1]][2].-Latt[x[2]][2]),(1,0)) , pairs)
    pairy = filter(x -> x[1] ∈ points2 && x[2] ∈ points2 && isequal(abs.(mod.(Latt[x[1]][2].-Latt[x[2]][2] .+ 1, Ly) .- 1), (0,1)) , pairs)
    return pairx,pairy
end

get_cellsize(Latt::CompositeLattice) = map(x -> maximum([Latt.subLatts[1].sites[ii][x] for ii in 1:div(size(Latt),length(Latt.subLatts))]),1:2)
get_cellsize(Latt::SimpleLattice) = map(x -> maximum([Latt.sites[ii][x] for ii in 1:size(Latt)]),1:2)


