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

function YCRect(L::Int64, W::Int64, (a,b)::NTuple{2,Float64} = (1.0,1.0),θ::Real = 0.0)
    @assert L ≥ W
    e = ((a, 0.0), (0.0, b))
    sites = [(x, y) for x in 1:L for y in 1:W]
    if iszero(θ)
         BC = PeriodicBoundaryCondition((0, W))
    else
         BC = TwistBoundaryCondition((0, W), θ)
    end
    return SquareLattice(e, sites, BC)
end

function ZZHoneyComb(L::Int64,W::Int64)
    shift = ((-1/2sqrt(3),1/2),(0.0,0.0),(1/sqrt(3),0.0),(sqrt(3)/2,1/2))
    return CompositeLattice([YCRect(L,W,(sqrt(3),1.0)) for _ in 1:4]..., shift) |> Snake!    
end
