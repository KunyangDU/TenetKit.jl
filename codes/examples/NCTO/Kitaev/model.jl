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
    J1xy,J1z,Jpm,Jzpm,
    J2 = 0.,J3xy = 0.,J3z = 0.,
    # gx = 4.8, gy = 4.85, gz = 2.5, 
    # μB = 0.05788,
    gx = 1, gy = 1, gz = 1, 
    μB = 1,
    Hx = 0, Hy = 0, Hz = 0, 
    hx = 0, hy = 0, hz = 0,kwargs...)

    LocalSpace = TrivialSpinOneHalf

    Root = InteractionTreeNode()
    triavals = get(kwargs,:triavals,[(1,0),(-1/2,sqrt(3)/2),(-1/2,-sqrt(3)/2)])
    direction = get(kwargs,:direction,[[0,1],[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2]])
    shift = get(kwargs,:shift, [0,1])
    bonds = getxyzbonds(Latt;shift = shift, direction=direction)
    for pair in neighbor(Latt;level = 1)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J1xy,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J1xy,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J1z,nothing)

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
            addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),Jpm_xx,nothing)
            addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),-Jpm_xx,nothing)
            addIntr!(Root,LocalSpace.SxSy,pair,("Sx","Sy"),(false,false),Jpm_xy,nothing)
            addIntr!(Root,LocalSpace.SySx,pair,("Sy","Sx"),(false,false),Jpm_xy,nothing)
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
            addIntr!(Root,LocalSpace.SxSz,pair,("Sx","Sz"),(false,false),Jzpm_xz,nothing)
            addIntr!(Root,LocalSpace.SzSx,pair,("Sz","Sx"),(false,false),Jzpm_xz,nothing)
            addIntr!(Root,LocalSpace.SySz,pair,("Sy","Sz"),(false,false),Jzpm_yz,nothing)
            addIntr!(Root,LocalSpace.SzSy,pair,("Sz","Sy"),(false,false),Jzpm_yz,nothing)
        end
    end

    for pair in neighbor(Latt;level = 2)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J2,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J2,nothing)
    end

    for pair in neighbor(Latt;level = 3)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),(false,false),J3xy,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),(false,false),J3xy,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J3z,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sx,i,"Sx",false,-μB*gx*Hx,nothing)
        addIntr!(Root,LocalSpace.Sy,i,"Sy",false,-μB*gy*Hy,nothing)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",false,-μB*gz*Hz,nothing)
    end

    cind = div(size(Latt),2)
    addIntr!(Root,LocalSpace.Sx,cind,"Sx",false,-hx,nothing)
    addIntr!(Root,LocalSpace.Sy,cind,"Sy",false,-hy,nothing)
    addIntr!(Root,LocalSpace.Sz,cind,"Sz",false,-hz,nothing)

    Lx,Ly = get_cellsize(Latt)
    pinh = get(kwargs,:pinh,repeat([[0.,0.,0.],],4Ly))
    pinsites = get(kwargs,:pinsites,vcat(1:2Ly,size(Latt)-2Ly-1:size(Latt)))

    for (i,hv) in enumerate(pinh)
        addIntr!(Root,LocalSpace.Sx,pinsites[i],"Sx",false,-hv[1],nothing)
        addIntr!(Root,LocalSpace.Sy,pinsites[i],"Sy",false,-hv[2],nothing)
        addIntr!(Root,LocalSpace.Sz,pinsites[i],"Sz",false,-hv[3],nothing)
    end

    return AutomataSparseMPO((Root),size(Latt))  
        
end


