
RBASIS3 = [[1.,0.,0.],[1/2,sqrt(3)/2,0.],[0.,0.,1.]]
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
MFBZpoint = [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]]
dZZFBZpoint = map(x -> x/2,FBZpoint)

# RBASIS3 = [[sqrt(3)/2,1/2,0.],[sqrt(3)/2,-1/2,0.],[0.,0.,1.]]
# KBASIS2 = kbasis2(RBASIS3)
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
# MFBZpoint = [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]]
# dZZFBZpoint = map(x -> x/2,FBZpoint)

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

function calcSSfactor(Latt::AbstractLattice,k::Vector,data::Dict,points = 1:size(Latt))
    S = let 
        S = 0
        for i in points,j in points
            i >= j && continue
            R = coordinate(Latt,i) .- coordinate(Latt,j)
            # norm(R) > 2 && continue
            S += data[(i,j)] * 2 * cos(dot(k,R)) / size(Latt)
        end
        S += 1/4
        S
    end
    return S
end

function calcSSfactor(Latt::AbstractLattice,k::Vector,datas::Vector,points = 1:size(Latt))
    S = let 
        S = zeros(length(datas))
        for i in points,j in points
            i >= j && continue
            R = coordinate(Latt,i) .- coordinate(Latt,j)
            # norm(R) > 4 && continue
            S .+= map(data -> data[(i,j)] * 2 * cos(dot(k,R)) / size(Latt),datas)
        end
        S .+= 1/4
        S
    end
    return S
end


function getCorrMat(Latt::AbstractLattice,data::Dict,dataonsite = nothing;
    selected_point = 1:size(Latt),
    independent_data = Dict(((i,),) => 0 for i in 1:size(Latt)))
    # L = length(selected_point)
    # A = zeros(L,L)
    # for i in 1:L,j in i+1:L
    #     A[i,j] = data[(selected_point[i],selected_point[j])]
    # end
    A = zeros(size(Latt),size(Latt))
    for i in selected_point, j in selected_point
        i >= j && continue
        A[i,j] = data[((i,j),)] - independent_data[((i,),)]*independent_data[((j,),)]
    end
    A = A + A'
    if !isnothing(dataonsite)
        if typeof(dataonsite) <: Dict
            for i in 1:L 
                A[i,i] = dataonsite[((i,),)] - independent_data[((i,),)]^2
            end
        elseif typeof(dataonsite) <: Vector
            A .+= diagm(dataonsite)
        elseif typeof(dataonsite) <: Number 
            # A .+= diagm(dataonsite*ones(L))
            tmp = zeros(size(Latt))
            tmp[selected_point] .= 1
            A .+= diagm((dataonsite .- [independent_data[((i,),)]^2 for i in 1:size(Latt)]) .* tmp)
        end
    end
    return A
end

function getCorrMat_old(Latt::AbstractLattice,data::Dict,dataonsite = nothing;selected_point = 1:size(Latt))
    L = length(selected_point)
    A = zeros(L,L)
    for i in 1:L,j in i+1:L
        A[i,j] = data[(selected_point[i],selected_point[j])]
    end
    A = A + A'
    if !isnothing(dataonsite)
        if typeof(dataonsite) <: Dict
            for i in 1:L 
                A[i,i] = dataonsite[((i,),)]
            end
        elseif typeof(dataonsite) <: Vector
            A .+= diagm(dataonsite)
        elseif typeof(dataonsite) <: Number 
            # A .+= diagm(dataonsite*ones(L))
            tmp = zeros(size(Latt))
            tmp[selected_point] .= 1
            A .+= diagm(dataonsite*tmp)
        end
    end
    return A
end

rbasis223(e::Tuple) = push!(collect(map(x -> vcat(x...,0),e)),[0,0,1])

