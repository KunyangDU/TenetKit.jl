
function tsvd(A::DenseMPOTensor{4}; direction::Symbol=:center, index_tuple = ((1,2,4),(3,)), kwargs...)
    @assert direction in [:center,:left,:right]
    trunc = get(kwargs,:trunc,TruncationScheme())[:]
    if direction == :center 
        U,S,V,ϵ = tsvd(A.A,index_tuple...;trunc = trunc)
        return map(DenseMPOTensor, (U,S,V))...,ϵ^2,BondInfo(S)
    elseif direction == :left 
        U,S,V,ϵ = tsvd(A.A,(2,),(1,3,4);trunc = trunc)
        return map(DenseMPOTensor,(U*S,permute(V,(2,1),(3,4))))...,ϵ^2,BondInfo(S)
    elseif direction == :right 
        U,S,V,ϵ = tsvd(A.A,(1,2,4),(3,);trunc = trunc)
        return map(DenseMPOTensor,(permute(U,(1,2),(4,3)),S*V))...,ϵ^2,BondInfo(S)
    end
end

function tsvd(A::CompositeMPOTensor{2,6}; direction::Symbol=:center, kwargs...)
    @assert direction in [:center,:left,:right]
    trunc = get(kwargs,:trunc,TruncationScheme())[:]
    U,S,V,ϵ = tsvd(A.A,(2,3,6),(1,4,5);trunc = trunc)
    # d = sqrt(@tensor S[1,2] * S'[2,1])
    # d != 0 && (ϵ /= d;vns = vonNeumann(S))
    if direction == :center
        return map(DenseMPOTensor,[permute(U,(1,2),(4,3)),S,permute(V,(2,1),(3,4))])...,ϵ^2,BondInfo(S)
    elseif direction == :left 
        return map(DenseMPOTensor,(permute(U*S,(1,2),(4,3)),permute(V,(2,1),(3,4))))...,ϵ^2,BondInfo(S)
    elseif direction == :right 
        return map(DenseMPOTensor,(permute(U,(1,2),(4,3)),permute(S*V,(2,1),(3,4))))...,ϵ^2,BondInfo(S)
    end
end

function leftorth!(obj::T,site::Int64) where T <: Union{DenseMPO, AdjointMPO}
    obj[site:site+1] = collect(leftorth(obj[site:site+1]...))
end
function rightorth!(obj::T,site::Int64) where T <: Union{DenseMPO, AdjointMPO}
    obj[site-1:site] = collect(rightorth(obj[site-1:site]...))
end

function leftorth(elm::DenseMPOTensor{4})
    Q,R = leftorth(elm.A,(1,2,4),(3,))
    return map(DenseMPOTensor,(permute(Q,(1,2),(4,3)),R))
end

function rightorth(A::DenseMPOTensor{4})
    L,Q = rightorth(A.A,(2,),(1,3,4))
    return map(DenseMPOTensor,(L,permute(Q,(2,1),(3,4))))
end

function leftorth(A::DenseMPOTensor{4}, B::DenseMPOTensor{4})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3 -4] ≔ Rm.A[-2,1]*B.A[-1,1,-3,-4]
    return Q,DenseMPOTensor(tmp)
end

function rightorth(A::DenseMPOTensor{4}, B::DenseMPOTensor{4})
    Lm,Q = rightorth(B)
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[-1,-2,1,-4]*Lm.A[1,-3]
    return DenseMPOTensor(tmp),Q
end

function leftorth(elm::AdjointMPOTensor{4})
    L,Q = rightorth(elm.A,(1,),(2,3,4))
    return map(AdjointMPOTensor,(permute(Q,(1,2),(3,4)),L))
end

function rightorth(A::AdjointMPOTensor{4})
    Q,R = leftorth(A.A,(1,2,3),(4,))
    return map(AdjointMPOTensor,(R,permute(Q,(1,2),(3,4))))
end

function leftorth(A::AdjointMPOTensor{4}, B::AdjointMPOTensor{4})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3 -4] ≔ Rm.A[1,-4]*B.A[-1,-2,-3,1]
    return Q,AdjointMPOTensor(tmp)
end

function rightorth(A::AdjointMPOTensor{4}, B::AdjointMPOTensor{4})
    Lm,Q = rightorth(B)
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[1,-2,-3,-4]*Lm.A[-1,1]
    return AdjointMPOTensor(tmp),Q
end


