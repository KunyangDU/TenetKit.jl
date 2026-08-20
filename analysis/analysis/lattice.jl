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
    # @assert L ≥ W
    e = ((a, 0.0), (0.0, b))
    sites = [(x, y) for x in 1:L for y in 1:W]
    if iszero(θ)
         BC = PeriodicBoundaryCondition((0, W))
    else
         BC = TwistBoundaryCondition((0, W), θ)
    end
    return SquareLattice(e, sites, BC)
end

function OpenRect(L::Int64, W::Int64, (a,b)::NTuple{2,Float64} = (1.0,1.0))
    #  L < W && @warn "L = $(L) < $(W) = W?"
     e = ((a, 0.0), (0.0, b))
     sites = [(x, y) for x in 1:L for y in 1:W]
     return SquareLattice(e, sites)
end

function OpenZZHoneyComb(L::Int64,W::Int64)
    shift₀ = [[-1/2sqrt(3),1/2],[0.0,0.0],[1/sqrt(3),0.0],[sqrt(3)/2,1/2]]
    shift = vcat([map(x -> x + (i-1)*[0.0,1.0],shift₀) for i in 1:W]...,map(x -> x .+ W*[0.0,1.0],[[0.0,0.0],[1/sqrt(3),0.0]]))
    return CompositeLattice([OpenRect(L,1,(sqrt(3),W*1.0)) for _ in 1:4W+2]..., Tuple(Tuple.(shift))) |> Snake!  
end

function OpenXCHoneyComb(L::Int64,W::Int64)
    shift₀ = [[1/2,-1/2sqrt(3)],[0.0,0.0],[0.0,1/sqrt(3)],[1/2,sqrt(3)/2]]
    shift = vcat([map(x -> x + (i-1)*[1.0,0.0],shift₀) for i in 1:L]...,map(x -> x .+ L*[1.0,0.0],[[0.0,0.0],[0.0,1/sqrt(3)]]))
    return CompositeLattice([OpenRect(1,W,(L*1.0,sqrt(3))) for _ in 1:4L+2]..., Tuple(Tuple.(shift))) |> Snake!  
end

function DiamondOpenXCHoneyComb(L::Int64,W::Int64)
    @assert W == 1
    shift₀ = [[1/2,-sqrt(3)/2],[1/2,-1/2sqrt(3)],[0.0,0.0],[0.0,1/sqrt(3)],[1/2,sqrt(3)/2],[1/2,5sqrt(3)/6]]
    shift₁ = [[1,-2sqrt(3)/3],[1,sqrt(3)]]
    shift = vcat([map(x -> x + (i-1)*[1.0,0.0],shift₀) for i in 1:L]...,[map(x -> x + (i-1)*[1.0,0.0],shift₁) for i in 1:L-1]...,map(x -> x .+ L*[1.0,0.0],[[0.0,0.0],[0.0,1/sqrt(3)]]))
    shift = map(x -> x - [L,0],shift)
    return CompositeLattice([OpenRect(1,W,(L*1.0,sqrt(3))) for _ in 1:8L]..., Tuple(Tuple.(shift))) |> Snake!  
end

function PCTria(L::Int64, W::Int64;
     scale::Real = 1.0)
    #  @assert L ≥ W
     # generic zigzag! implementation can work if using the following convention
     e = ((1.,0.).*scale, (1/2,sqrt(3)/2).*scale)
     sites = [(x, y) for x in 1:L for y in 1:W]
     BC = PeriodicBoundaryCondition((0, W))
     return TriangularLattice(e, sites, BC)
end


function PCHoneyComb(Lx::Int64, Ly::Int64)
    shift = ((0.0,0.0),(1/2,sqrt(3)/6))
    return CompositeLattice([PCTria(Lx,Ly) for _ in 1:2]..., shift) |> Snake! 
end

function ZZHoneyComb(L::Int64,W::Int64)
    shift = ((-1/2sqrt(3),1/2),(0.0,0.0),(1/sqrt(3),0.0),(sqrt(3)/2,1/2))
    return CompositeLattice([YCRect(L,W,(sqrt(3),1.0)) for _ in 1:4]..., shift) |> Snake!    
end

function ACHoneyComb(L::Int64,W::Int64)
    shift = ((1/2,-1/2sqrt(3)),(0.0,0.0),(0.0,1/sqrt(3)),(1/2,sqrt(3)/2))
    return CompositeLattice([YCRect(L,W,(1.0,sqrt(3))) for _ in 1:4]..., shift) |> Snake!    
end

function YCHoneyComb(Lx::Int64, Ly::Int64)
    shift = ((0.0,0.0),(sqrt(3)/6,1/2))
    return CompositeLattice([YCTria(Lx,Ly) for _ in 1:2]..., shift) |> Snake! 
end

# function XCHoneyComb(Lx::Int64, Ly::Int64)
#     shift = ((0.0,0.0),(1/2, sqrt(3)/6))
#     return CompositeLattice([XCTria(Lx,Ly) for _ in 1:2]..., shift) |> Snake!   
# end

function XCHoneyComb(L::Int64,W::Int64)
    shift = ((1/2,-1/2sqrt(3)),(0.0,0.0),(0.0,1/sqrt(3)),(1/2,sqrt(3)/2))
    return CompositeLattice([YCRect(L,W,(1.0,sqrt(3))) for _ in 1:4]..., shift) |> Snake!    
end

function _OpenSqua(L::Int64, W::Int64, a = (1,1))
    # L < W && @warn "L = $(L) < $(W) = W?"
    e = ((a[1], 0.0), (0.0, a[2]))
    sites = [(x, y) for x in 1:L for y in 1:W]
    return SquareLattice(e, sites)
end

# function OXCHoneyComb(L::Int64,W::Int64)
#     shift = ((-1/2sqrt(3),1/2),(0.0,0.0),(1/sqrt(3),0.0),(sqrt(3)/2,1/2))
#     return CompositeLattice([_OpenSqua(L,W,(sqrt(3),1.0)) for _ in 1:4]..., shift) |> Snake!    
# end

# function OpenXCHoneyComb(L::Int64,W::Int64)
#     shift = ((1/2,-1/2sqrt(3)),(0.0,0.0),(0.0,1/sqrt(3)),(1/2,sqrt(3)/2))
#     return CompositeLattice([_OpenSqua(L,W,(1.0,sqrt(3))) for _ in 1:4]..., shift) |> Snake!    
# end


# function getxyzbonds(Latt::AbstractLattice;
#     shift = [0,1],
#     direction = [[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]],tol=1e-8)
#     nb = neighbor(Latt)
#     _,Ly = get_cellsize(Latt)
#     return map(direction) do v
#         filter(x -> abs(dot(let 
#             u = coordinate(Latt,x[1]) .- coordinate(Latt,x[2])
#             if abs(u[2]) > 1
#                 u = u .- sign(u[2])*shift*Ly
#             end
#             u
#         end,v)) < tol ,nb)
#     end
# end

function getxyzbonds(Latt::AbstractLattice,direction::Vector,tol::Float64=1e-8)
    nb = neighbor(Latt)
    _,Ly = get_cellsize(Latt)
    return map(direction) do v
        filter(x -> abs(dot(relaVec(Latt,x...),v)) < tol ,nb)
    end
end

function _OHTria(L::Int64,e::Tuple;kwargs...)
    scale = get(kwargs,:scale,1)
    neglect_sites = get(kwargs,:neglect_sites,[])
    sites = [(x, y) for x in 0:L-1 for y in 0:L-1-x]
    select_sites = get(kwargs,:select_sites,sites)
    path = get(kwargs,:path,Snake!)
    e = map(x -> x .* scale,e)
    N = length(select_sites)
    for _ in 1:5 
        sites_rotate = map(select_sites[end - N + 1: end]) do (x, y)
            return (-y, x + y)  
        end
        append!(select_sites, sites_rotate)
    end
    unique!(select_sites)
    filter!(x -> x ∉ neglect_sites, select_sites)
    return TriangularLattice(e, select_sites) |> path
end

function OHHoneyComb(L::Int64;kwargs...)
    cLatt = _OHTria(L,((1.0, 0.0),(1/2, sqrt(3)/2));scale = sqrt(3))
    Latt = OHTria(2L)
    # Latt = cLatt
    csites = map( y -> findmin(x -> FiniteLattices.norm(coordinate(Latt,x) .- coordinate(cLatt,y)),1:size(Latt))[2],1:size(cLatt))
    hcsites = vcat(map(x -> map(y -> y[1] == x ? y[2] : y[1], neighbor(Latt,x)),csites)...)
    unique!(hcsites)
    hcsites = map(x -> Latt[x],hcsites)
    return _OHTria(2L,((sqrt(3)/2, 1/2),(0.0, 1.0));select_sites = hcsites,kwargs...)
end

function getFlux(Latt::AbstractLattice, flux_Latt::AbstractLattice, direction::Vector =[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]];d::Number = 1)
    fluxsites = map(y -> Tuple(sort(filter(x -> abs(norm(coordinate(Latt,x) .- coordinate(flux_Latt,y)) - d) < 1e-8,1:size(Latt)))), 1:size(flux_Latt))
    fluxdirections = []
    for flux in fluxsites
        tmpdirection = Int64[]
        for i in flux
            nbp = intersect(unique(vcat(collect.(neighbor(Latt,i))...)),filter(x -> x != i,flux))
            pv = coordinate(Latt,nbp[1]) .+ coordinate(Latt,nbp[2]) .- 2 .* coordinate(Latt,i)
            push!(tmpdirection,findmin(x -> abs(dot(x,pv)),direction)[2])
        end
        push!(fluxdirections, Tuple(tmpdirection))
    end
    return fluxsites,fluxdirections,direction
end


function getPBCflux(Latt::AbstractLattice, flux_Latt::AbstractLattice, direction::Vector =[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]];
    d::Number = 1,edge_shift::Vector = [0,0],flux_shift::Vector = [0,0])
    Lx,Ly = get_cellsize(Latt)
    fluxsites = map(y -> Tuple(sort(filter(x -> abs(norm(coordinate(Latt,x) .- coordinate(flux_Latt,y) .- flux_shift) - d) < 1e-8 || abs(norm(coordinate(Latt,x) .+ Ly .* edge_shift .- coordinate(flux_Latt,y) .- flux_shift) - d) < 1e-8,1:size(Latt)))),1:size(flux_Latt))
    fluxdirections = []
    for flux in fluxsites
        tmpdirection = Int64[]
        for i in flux
            nbp = intersect(unique(vcat(collect.(neighbor(Latt,i))...)),filter(x -> x != i,flux))
            pv1 = coordinate(Latt,nbp[1]) .- coordinate(Latt,i)
            pv2 = coordinate(Latt,nbp[2]) .- coordinate(Latt,i)
            abs(pv1[2]) > edge_shift[2] && (pv1 = pv1 .- sign(pv1[2])*edge_shift*Ly)
            abs(pv2[2]) > edge_shift[2] && (pv2 = pv2 .- sign(pv2[2])*edge_shift*Ly)
            pv = pv1 .+ pv2
            push!(tmpdirection,findmin(x -> abs(dot(x,pv)),direction)[2])
        end
        push!(fluxdirections, Tuple(tmpdirection))
    end
    return fluxsites,fluxdirections,direction
end

function getOBCflux(Latt::AbstractLattice, flux_Latt::AbstractLattice, direction::Vector ;
    d::Number = 1,total_shift::Vector = [0,0])
    fluxsites = map(y -> Tuple(sort(filter(x -> abs(norm(coordinate(Latt,x) .- coordinate(flux_Latt,y) .- total_shift) - d) < 1e-8,1:size(Latt)))),1:size(flux_Latt))
    # filter!(x -> length(x) == 6,fluxsites)
    fluxdirections = []
    for flux in fluxsites
        if length(flux) ≠ 6
            push!(fluxdirections,())
            continue
        end
        tmpdirection = Int64[]
        for i in flux
            nbp = intersect(unique(vcat(collect.(neighbor(Latt,i))...)),filter(x -> x != i,flux))
            pv1 = coordinate(Latt,nbp[1]) .- coordinate(Latt,i)
            pv2 = coordinate(Latt,nbp[2]) .- coordinate(Latt,i)
            pv = pv1 .+ pv2
            push!(tmpdirection,findmin(x -> abs(dot(x,pv)),direction)[2])
        end
        push!(fluxdirections, Tuple(tmpdirection))
    end
    return fluxsites,fluxdirections,direction
end