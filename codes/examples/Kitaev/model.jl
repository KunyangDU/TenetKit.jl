# RBASIS3 = [[sqrt(3)/2,1/2,0.],[sqrt(3)/2,-1/2,0.],[0.,0.,1.]]
# KBASIS2 = kbasis2(RBASIS3)
# # KBASIS2 = [ (3.6275987284684357, 6.283185307179586), (3.6275987284684357, -6.283185307179586)]
# RBASIS2 = map(x -> x[1:2],RBASIS3[1:2])
# KitaevBasis2 = [[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]
# FBZpoint = [[1/3,2/3],[2/3,1/3],[1/3,-1/3],[-1/3,-2/3],[-2/3,-1/3],[-1/3,1/3]]
# BASIS2 = [[1.,0.],[0.,1.]]
# Triapoint = [[-sqrt(3)/2,-1/2],[sqrt(3)/2,-1/2],[0.,1]]
# JBASIS2 = Triapoint
# KitaevBDpoiont = let 
#     a,b,c = JBASIS2
#     [(a+b)/2,(b+c)/2,(a+c)/2]
# end

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
    K::Number = 1, ϵ::Number = 0.0,
    Hx::Number = 0.0, Hy::Number = 0.0, Hz::Number = 0.0,
    root::Bool = false,
    kwargs...)

    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()
    
    shift = get(kwargs,:shift,[0,1])
    direction = get(kwargs,:direction,[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]])

    xbonds,ybonds,zbonds = getxyzbonds(Latt;shift = shift,direction = direction)
    
    for pair in xbonds
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),K + ϵ,nothing)
    end
    for pair in ybonds
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),K + ϵ,nothing)
    end
    for pair in zbonds
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),K + ϵ,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sx,i,"Sx",false,-Hx,nothing)
        addIntr!(Root,LocalSpace.Sy,i,"Sy",false,-Hy,nothing)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",false,-Hz,nothing)
    end

    return root ? Root : AutomataSparseMPO(Root,size(Latt))
end
