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
MFBZpoint = [[1,0],[1,1],[0,1],[-1,0],[-1,-1],[0,-1]]
dZZFBZpoint = map(x -> x/2,FBZpoint)


function getxyzbonds(Latt::AbstractLattice;
    shift = [0,1],
    direction = [[1,0],[1/2,-sqrt(3)/2],[1/2,sqrt(3)/2]],
    projection = sqrt(3)/3
    )

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

