# function getxyzbonds(Latt::AbstractLattice;
#     shift = [0,1],
#     direction = [[1,0],[1/2,-sqrt(3)/2],[1/2,sqrt(3)/2]],
#     projection = sqrt(3)/3
#     )

#     nb = neighbor(Latt)
#     _,Ly = get_cellsize(Latt)
#     return map(direction) do v
#         filter(x -> abs(dot(let 
#             u = coordinate(Latt,x[1]) .- coordinate(Latt,x[2])
#             if abs(u[2]) > 1
#                 u = u .- sign(u[2])*shift*Ly
#             end
#             u
#         end,v)) ≈ projection ,nb)
#     end
# end

R1 = [
    cos(pi/4) -sin(pi/4) 0;
    sin(pi/4) cos(pi/4) 0;
    0 0 1
]

R2 = [
    1 0 0;
    0 sqrt(3)/3 -sqrt(6)/3;
    0 sqrt(6)/3 sqrt(3)/3
]

Py2c = R2*R1

PY2C = [
    2/3 1/3 2/3 -sqrt(2)/3;
    0 0 -2 sqrt(2);
    -1/3 1/3 -4/3 -sqrt(2)/3;
    -1/3 1/3 2/3 sqrt(2)/6
]
PC2Y = inv(PY2C)

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

function _x_stripe_points_XCHC(Lx::Int64,Ly::Int64,shift::Int64 = 0)
    if iseven(shift)
        mshift = mod(div(shift,2), div(Ly,2))
        return vcat(1 + mshift, [((2i+1)Ly + div(Ly,2) - mshift*2) .+ (0:1)*(1 + mshift*4) for i in 0:(Lx-2)]..., size(Latt) - div(Ly,2) - mshift*2)
    else
        mshift = mod(div(shift-1,2), div(Ly,2)) + 1
        return vcat([((2i+1)Ly + div(Ly,2) - mshift*2 + 1) .+ (0:1)*(4*mshift - 1) for i in 0:Lx-2]...,size(Latt) - div(Ly,2) - mshift*2 + 1, size(Latt) - div(Ly,2) + mshift)
    end
end


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

function PCTria(L::Int64, W::Int64;
     scale::Real = 1.0)
     @assert L ≥ W
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

function XCHoneyComb(Lx::Int64, Ly::Int64)
    shift = ((0.0,0.0),(1/2, sqrt(3)/6))
    return CompositeLattice([XCTria(Lx,Ly) for _ in 1:2]..., shift) |> Snake!   
end

function TrivialHamiltonian(Latt::AbstractLattice;
    K::Number = 1.0,Γ::Number = 0.0,Γ′::Number = 0.0,J::Number = 0.0,
    Hx = 0, Hy = 0, Hz = 0,kwargs...)

    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()
    triavals = get(kwargs,:triavals,[(1,0),(-1/2,sqrt(3)/2),(-1/2,-sqrt(3)/2)])
    direction = get(kwargs,:direction,[[0,1],[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2]])
    shift = get(kwargs,:shift, [0,1])
    bonds = getxyzbonds(Latt;shift = shift, direction=direction)

    for pair in bonds[1]
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J + K,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J,nothing)

        addIntr!(Root,LocalSpace.SySz,pair,("Sy","Sz"),(false,false),Γ,nothing)
        addIntr!(Root,LocalSpace.SzSy,pair,("Sz","Sy"),(false,false),Γ,nothing)
        addIntr!(Root,LocalSpace.SxSz,pair,("Sx","Sz"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SzSx,pair,("Sz","Sx"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SySx,pair,("Sy","Sx"),(false,false),Γ′,nothing)
    end

    for pair in bonds[2]
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J + K,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J,nothing)

        addIntr!(Root,LocalSpace.SzSx,pair,("Sz","Sx"),(false,false),Γ,nothing)
        addIntr!(Root,LocalSpace.SxSz,pair,("Sx","Sz"),(false,false),Γ,nothing)
        addIntr!(Root,LocalSpace.SySz,pair,("Sy","Sz"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SzSy,pair,("Sz","Sy"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SySx,pair,("Sy","Sx"),(false,false),Γ′,nothing)
    end

    for pair in bonds[3]
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J + K,nothing)

        addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),(false,false),Γ,nothing)
        addIntr!(Root,LocalSpace.SySx,pair,("Sx","Sy"),(false,false),Γ,nothing)
        addIntr!(Root,LocalSpace.SxSz,pair,("Sx","Sz"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SzSx,pair,("Sz","Sx"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SzSy,pair,("Sz","Sy"),(false,false),Γ′,nothing)
        addIntr!(Root,LocalSpace.SySz,pair,("Sy","Sz"),(false,false),Γ′,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sx,i,"Sx",false,-Hx,nothing)
        addIntr!(Root,LocalSpace.Sy,i,"Sy",false,-Hy,nothing)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",false,-Hz,nothing)
    end
    if get(kwargs,:returnnode,false)
        return Root
    else
        return AutomataSparseMPO((Root),size(Latt))
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
    fluxsites = map(y -> Tuple(sort(filter(x -> abs(norm(coordinate(Latt,x) .- coordinate(flux_Latt,y) .- flux_shift) - d) < 1e-8 || abs(norm(coordinate(Latt,x) .+ Ly .* edge_shift .- coordinate(flux_Latt,y) .- flux_shift) - d) < 1e-8,1:size(Latt)))),1:size(flux_Latt))
    fluxdirections = []
    for flux in fluxsites
        tmpdirection = Int64[]
        for i in flux
            nbp = intersect(unique(vcat(collect.(neighbor(Latt,i))...)),filter(x -> x != i,flux))
            pv1 = coordinate(Latt,nbp[1]) .- coordinate(Latt,i)
            pv2 = coordinate(Latt,nbp[2]) .- coordinate(Latt,i)
            abs(pv1[2]) > 1 && (pv1 = pv1 .- sign(pv1[2])*edge_shift*Ly)
            abs(pv2[2]) > 1 && (pv2 = pv2 .- sign(pv2[2])*edge_shift*Ly)
            pv = pv1 .+ pv2
            push!(tmpdirection,findmin(x -> abs(dot(x,pv)),direction)[2])
        end
        push!(fluxdirections, Tuple(tmpdirection))
    end
    return fluxsites,fluxdirections,direction
end
