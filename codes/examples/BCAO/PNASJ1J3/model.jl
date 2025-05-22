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

function TrivialHamiltonian(Latt::AbstractLattice;
    J1xy::Float64,J1z::Float64,D::Float64,E::Float64,
    J3xy::Float64,J3z::Float64,
    gx = 4.8, gy = 4.85, gz = 2.5, μB = 0.05788,
    Hx = 0, Hy = 0, Hz = 0,
    hx = 0, hy = 0, hz = 0
    )
    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()
    # triavals = [(1,0),(-1/2,sqrt(3)/2),(-1/2,-sqrt(3)/2)]
    bonds = getxyzbonds(Latt)

    Az = [
        J1xy+D E 0.;
        E J1xy-D 0.;
        0. 0. J1z
    ]
    U = [
        cos(2pi/3) -sin(2pi/3) 0.;
        sin(2pi/3) cos(2pi/3) 0.;
        0. 0. 1.
    ]
    
    Ay = U'*Az*U
    Ax = U*Az*U'
    As = [Ax,Ay,Az]

    for pair in neighbor(Latt;level = 1)
        for i in 1:3
            if pair in bonds[i]
                A = As[i]
                addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),A[1,1],nothing)
                addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),A[2,2],nothing)
                addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),A[3,3],nothing)

                addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),A[1,2],nothing)
                addIntr!(Root,LocalSpace.SySx,pair,("Sy","Sx"),A[2,1],nothing)
            end
        end
    end

    for pair in neighbor(Latt;level = 3)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J3xy,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J3xy,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J3z,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sx,i,"Sx",-μB*gx*Hx,nothing)
        addIntr!(Root,LocalSpace.Sy,i,"Sy",-μB*gy*Hy,nothing)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",-μB*gz*Hz,nothing)
    end

    cind = div(size(Latt),2)
    addIntr!(Root,LocalSpace.Sx,cind,"Sx",-hx,nothing)
    addIntr!(Root,LocalSpace.Sy,cind,"Sy",-hy,nothing)
    addIntr!(Root,LocalSpace.Sz,cind,"Sz",-hz,nothing)

    return AutomataSparseMPO(InteractionTree(Root),size(Latt))
end


