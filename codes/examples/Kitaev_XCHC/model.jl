function XCHC(L::Int64,W::Int64)
    shift = ((0.0,0.0),(0.,1/sqrt(3)))
    return CompositeLattice([XCTria(L,W) for _ in 1:2]..., shift) |> Snake!
end


function getxyzbonds(Latt::AbstractLattice;
    shift = [0,sqrt(3)/2],
    direction = [[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]],tol=1e-8)
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

function Hamiltonian(Latt::AbstractLattice;Jx::Number = 1,Jy::Number = 1, Jz::Number = 1)

    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()

    xbonds,ybonds,zbonds = getxyzbonds(Latt)
    # @show length.([xbonds,ybonds,zbonds])
    
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

