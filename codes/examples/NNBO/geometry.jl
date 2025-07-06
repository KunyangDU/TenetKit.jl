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
Pc2y = Py2c'
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

function _Cub2Cry(params_cubic::NamedTuple;P::Matrix = PC2Y)
    v = collect(params_cubic)
    v1 = P*v
    return (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

# _Cub2Cry(A::Matrix;P::AbstractMatrix = Pc2y) = P'*A*P
# _Cub2Cry(A::Vector;P::AbstractMatrix = Pc2y) = [_Cub2Cry(x;P=P) for x in A]

function _One2OneHalf1(A::AbstractMatrix,D::Number)
    Jxx = -((A[1,2] + A[2,2])^2-(A[1,2] - A[2,1])^2)/16/D
    Jzz = A[3,3] + ((A[1,2] + A[2,2])^2+(A[1,2] - A[2,1])^2)/16/D
    Jxy = ((A[1,2] + A[2,2])*(A[1,2] - A[2,1]))/4/D
    Jyx = -((A[1,2] + A[2,2])*(A[1,2] - A[2,1]))/4/D
    return (Jxx = Jxx, Jyy = Jxx, Jxy = Jxy, Jyx = Jyx, Jzz = Jzz)
end
