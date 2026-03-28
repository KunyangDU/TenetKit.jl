function addIntr2(inds::NTuple{2,Int64},Ops::Tuple,L::Int64,strength::Number = 1)
    strength ≈ 0 && return zeros(size(Ops[1])[1]^L,size(Ops[1])[1]^L)
    id = diagm(ones(size(Ops[1])[1]))
    idl,idr = inds
    tot = idl == 1 ? Ops[1] : id 
    for i in 2:L
        if i == idl
            tot = kron(tot,Ops[1])
        elseif i == idr
            tot = kron(tot,Ops[2])
        else
            tot = kron(tot,id)
        end
    end
    return tot * strength
end

function addIntr6(inds::NTuple{6,Int64},Ops::Tuple,L::Int64,strength::Number = 1)
    strength ≈ 0 && return zeros(size(Ops[1])[1]^L,size(Ops[1])[1]^L)
    inds = collect(inds)
    Ops = collect(Ops)
    id = diagm(ones(size(Ops[1])[1]))
    if inds[1] == 1
        tot = popfirst!(Ops)
        popfirst!(inds)
    else
        tot = id 
    end
    for i in 2:L
        if isempty(inds) || i ≠ inds[1]
            tot = kron(tot,id)
        else
            tot = kron(tot,popfirst!(Ops))
            popfirst!(inds)
        end
    end
    return tot * strength
end

function addIntr1(ind::Int64,Op::AbstractMatrix,L::Int64,strength::Number = 1)
    strength ≈ 0 && return zeros(size(Op)[1]^L,size(Op)[1]^L)
    id = diagm(ones(size(Op)[1]))
    tot = ind == 1 ? Op : id 
    for i in 2:L
        if i == ind
            tot = kron(tot,Op)
        else
            tot = kron(tot,id)
        end
    end
    return tot * strength
end

get_cellsize(Latt::CompositeLattice) = map(x -> maximum([Latt.subLatts[1].sites[ii][x] for ii in 1:div(size(Latt),length(Latt.subLatts))]),1:2)
get_cellsize(Latt::SimpleLattice) = map(x -> maximum([Latt.sites[ii][x] for ii in 1:size(Latt)]),1:2)

function diagm(A::Pair{Int64, Vector{T}}) where T
    L = abs(A.first) + length(A.second)
    B = zeros(T,L,L)
    for i in 1:length(A.second)
        B[(A.first > 0 ? (i,abs(A.first) + i) : (abs(A.first) + i,i))...] = A.second[i]
    end
    return B
end
diagm(A::Vector) = diagm(0 => A)

function getPBCflux(Latt::AbstractLattice, flux_Latt::AbstractLattice, direction::Vector =[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]];
    d::Number = 1,edge_shift::Vector = [0,0],flux_shift::Vector = [0,0])
    fluxsites = map(y -> Tuple(sort(filter(x -> abs(norm(coordinate(Latt,x) .- coordinate(flux_Latt,y) .- flux_shift) - d) < 1e-8 || abs(norm(coordinate(Latt,x) .+ Ly .* edge_shift .- coordinate(flux_Latt,y) .- flux_shift) - d) < 1e-8,1:size(Latt)))),1:size(flux_Latt))
    fluxdirections = []
    for flux in fluxsites
        tmpdirection = Int64[]
        for i in flux
            nbp = intersect(unique(vcat(collect.(neighbor(Latt,i))...)),filter(x -> x != i,flux))
            pv1 = coordinate(Latt,nbp[1]) .- coordinate(Latt,i)
            pv2 = coordinate(Latt,nbp[2]) .- coordinate(Latt,i)
            abs(pv1[2]) > 1 && (pv1 = pv1 .- sign(pv1[2])*edge_shift*Ly)
            abs(pv2[2]) > 1 && (pv2 = pv2 .- sign(pv2[2])*edge_shift*Ly)
            pv = pv1 .+ pv2
            push!(tmpdirection,findmin(x -> abs(dot(x,pv)),direction)[2])
        end
        push!(fluxdirections, Tuple(tmpdirection))
    end
    return fluxsites,fluxdirections,direction
end

