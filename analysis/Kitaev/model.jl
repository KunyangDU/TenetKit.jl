RBASIS3 = [[sqrt(3)/2,1/2,0.],[sqrt(3)/2,-1/2,0.],[0.,0.,1.]]
KBASIS2 = kbasis2(RBASIS3)
RBASIS2 = map(x -> x[1:2],RBASIS3[1:2])
KitaevBasis2 = [[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]
FBZpoint = [[1/3,2/3],[2/3,1/3],[1/3,-1/3],[-1/3,-2/3],[-2/3,-1/3],[-1/3,1/3]]
BASIS2 = [[1.,0.],[0.,1.]]
Triapoint = [[-sqrt(3)/2,-1/2],[sqrt(3)/2,-1/2],[0.,1]]
JBASIS2 = Triapoint
KitaevBDpoiont = let 
    a,b,c = JBASIS2
    [(a+b)/2,(b+c)/2,(a+c)/2]
end


function getxyzbonds(Latt::AbstractLattice;shift = [0,1],direction = [[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2],[1,0]],projection = sqrt(3)/3)
    nb = neighbor(Latt)
    _,Ly = get_cellsize(Latt)
    return map(direction) do v
        filter(x -> abs(dot(let 
            u = coordinate(Latt,x[1]) .- coordinate(Latt,x[2])
            if abs(u[2]) > 1
                u = u .- sign(u[2])*shift*Ly
            end
            u
        end,v)) ≈ projection ,nb)
    end
end

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

function TrivialHamiltonian(Latt::AbstractLattice;Jx::Number = 1,Jy::Number = 1, Jz::Number = 1)

    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()

    xbonds,ybonds,zbonds = getxyzbonds(Latt)
    
    for pair in xbonds
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),Jx,nothing)
    end
    for pair in ybonds
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),Jy,nothing)
    end
    for pair in zbonds
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),Jz,nothing)
    end

    # addIntr!(Root,LocalSpace.Sz,div(size(Latt),2),"Sz",h,nothing)

    return AutomataSparseMPO(InteractionTree(Root),size(Latt))  
        
end

function getKitaevSS(lsk,params::NamedTuple)
    return getKitaevSS(lsk,[params.Jx,params.Jy,params.Jz])
end

function getKitaevSS(lsk,params::Vector)
    ϵ(k1,k2,params) = 2(params[1]*cos(k1) + params[2]*cos(k2) + params[3])
    Δ(k1,k2,params) = 2(params[1]*sin(k1) + params[2]*sin(k2))

    S = 8*sqrt(3)*pi^2/3
    Nk = length(lsk)
    I = let 
        I = 0
        for k in lsk
            ks = map(x -> dot(k,x),KitaevBasis2)
            I += -(sqrt(3)/32/pi^2) * (S/Nk) * ϵ(ks...,params) / sqrt(ϵ(ks...,params)^2 + Δ(ks...,params)^2)
        end
        I
    end
    return I
end


