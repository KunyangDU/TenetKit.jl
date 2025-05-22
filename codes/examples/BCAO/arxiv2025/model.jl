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
    J1xy,J1z,J2,J3xy,J3z,Jpm,Jzpm,
    gx = 4.8, gy = 4.85, gz = 2.5, μB = 0.05788,
    Hx = 0, Hy = 0, Hz = 0, 
    hx = 0, hy = 0, hz = 0,kwargs...)

    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()
    triavals = [(1,0),(-1/2,sqrt(3)/2),(-1/2,-sqrt(3)/2)]
    direction = [[0,1],[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2]]
    bonds = getxyzbonds(Latt;direction=direction)
    for pair in neighbor(Latt;level = 1)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J1xy,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J1xy,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J1z,nothing)

        if Jpm != 0
            Jpm_xx = 0
            Jpm_xy = 0
            for j in 1:3
                if pair in bonds[j]
                    Jpm_xx = 2 * Jpm * triavals[j][1]
                    Jpm_xy = - 2 * Jpm * triavals[j][2]
                    break
                end
            end
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),Jpm_xx,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),-Jpm_xx,nothing)
            addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),Jpm_xy,nothing)
            addIntr!(Root,LocalSpace.SySx,pair,("Sy","Sx"),Jpm_xy,nothing)
        end

        if Jzpm != 0    
            Jzpm_xz = 0
            Jzpm_yz = 0
            for j in 1:3
                if pair in bonds[j]
                    Jzpm_xz = Jzpm * triavals[j][2]
                    Jzpm_yz = - Jzpm * triavals[j][1]
                    break
                end
            end
            addIntr!(Root,LocalSpace.SxSz,pair,("Sx","Sz"),Jzpm_xz,nothing)
            addIntr!(Root,LocalSpace.SzSx,pair,("Sz","Sx"),Jzpm_xz,nothing)
            addIntr!(Root,LocalSpace.SySz,pair,("Sy","Sz"),Jzpm_yz,nothing)
            addIntr!(Root,LocalSpace.SzSy,pair,("Sz","Sy"),Jzpm_yz,nothing)
        end
    end

    for pair in neighbor(Latt;level = 2)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J2,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J2,nothing)
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

    Lx,Ly = get_cellsize(Latt)
    pinh = get(kwargs,:pinh,repeat([[0.,0.,0.],],4Ly))
    pinsites = get(kwargs,:pinsites,vcat(1:2Ly,size(Latt)-2Ly-1:size(Latt)))

    for (i,hv) in enumerate(pinh)
        addIntr!(Root,LocalSpace.Sx,pinsites[i],"Sx",-hv[1],nothing)
        addIntr!(Root,LocalSpace.Sy,pinsites[i],"Sy",-hv[2],nothing)
        addIntr!(Root,LocalSpace.Sz,pinsites[i],"Sz",-hv[3],nothing)
    end

    return AutomataSparseMPO(InteractionTree(Root),size(Latt))  
        
end

