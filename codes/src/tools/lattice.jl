"""
YCSqua() without L ≥ W check
"""
function iYCSqua(L::Int64, W::Int64, θ::Real = 0.0)
    e = ((1.0, 0.0), (0.0, 1.0))
    sites = [(x, y) for x in 1:L for y in 1:W]
    if iszero(θ)
         BC = PeriodicBoundaryCondition((0, W))
    else
         BC = TwistBoundaryCondition((0, W), θ)
    end
    return SquareLattice(e, sites, BC)
end

function getPBCflux(Latt::AbstractLattice, flux_Latt::AbstractLattice, direction::Vector =[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]];
    d::Number = 1,edge_shift::Vector = [0,0],flux_shift::Vector = [0,0])
    Lx,Ly = get_cellsize(Latt)
    fluxsites = filter(x -> length(x) == 6,  map(y -> Tuple(sort(filter(x -> abs(norm(coordinate(Latt,x) .- coordinate(flux_Latt,y) .- flux_shift) - d) < 1e-8 || abs(norm(coordinate(Latt,x) .+ Ly .* edge_shift .- coordinate(flux_Latt,y) .- flux_shift) - d) < 1e-8,1:size(Latt)))),1:size(flux_Latt)))
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


