

function TensorKit.leftorth(elm::DenseMPOTensor{4})
    Q,R = leftorth(elm.A,(1,2,4),(3,))
    return map(DenseMPOTensor,(permute(Q,(1,2),(4,3)),R))
end

function TensorKit.leftorth!(A::DenseMPOTensor{4}, B::DenseMPOTensor{4})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3 -4] ≔ Rm.A[-2,1]*B.A[-1,1,-3,-4]
    A.A = Q.A
    B.A = tmp
end

function TensorKit.leftorth!(obj::DenseMPO,site::Int64)
    leftorth!(obj[site:site+1]...)
end

function TensorKit.rightorth(A::DenseMPOTensor{4})
    L,Q = rightorth(A.A,(2,),(1,3,4))
    return map(DenseMPOTensor,(L,permute(Q,(2,1),(3,4))))
end

function TensorKit.rightorth!(A::DenseMPOTensor{4}, B::DenseMPOTensor{4})
    Lm,Q = rightorth(B)
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[-1,-2,1,-4]*Lm.A[1,-3]
    A.A = tmp
    B.A = Q.A
end

function TensorKit.rightorth!(obj::DenseMPO,site::Int64)
    rightorth!(obj[site-1:site]...)
end

function TensorKit.tsvd(A::CompositeMPOTensor{2,6}; direction::Symbol=:center, kwargs...)
    @assert direction in [:center,:left,:right]
    vns = nothing
    U,S,V,ϵ = tsvd(A.A,(2,3,6),(1,4,5);kwargs...)
    d = sqrt(@tensor S[1,2] * S'[2,1])

    if direction == :center
        d != 0 && (ϵ /= d)
        return map(DenseMPOTensor,[permute(U,(1,2),(4,3)),S,permute(V,(2,1),(3,4))])...,ϵ^2
    elseif direction == :left 
        d = sqrt(@tensor S[1,2] * S'[2,1])
        if d != 0
            ϵ /= d
            vns = vonNeumann(S)
        end
        return map(DenseMPOTensor,(permute(U*S,(1,2),(4,3)),permute(V,(2,1),(3,4))))...,ϵ^2,vns
    elseif direction == :right 
        d = sqrt(@tensor S[1,2] * S'[2,1])
        if d != 0
            ϵ /= d
            vns = vonNeumann(S)
        end
        return map(DenseMPOTensor,(permute(U,(1,2),(4,3)),permute(S*V,(2,1),(3,4))))...,ϵ^2,vns
    end
end

function TensorKit.tsvd(A::DenseMPOTensor{4}; direction::Symbol=:center, index_tuple = ((1,2,4),(3,)), kwargs...)
    @assert direction in [:center,:left,:right]
    if direction == :center 
        U,S,V,ϵ = tsvd(A.A,index_tuple...;kwargs...)
        return map(DenseMPOTensor, (U,S,V))...,ϵ^2
    elseif direction == :left 
        U,S,V,ϵ = tsvd(A.A,(2,),(1,3,4);kwargs...)
        return map(DenseMPOTensor,(U*S,permute(V,(2,1),(3,4))))...,ϵ^2
    elseif direction == :right 
        U,S,V,ϵ = tsvd(A.A,(1,2,4),(3,);kwargs...)
        return map(DenseMPOTensor,(permute(U,(1,2),(4,3)),S*V))...,ϵ^2
    end
end


