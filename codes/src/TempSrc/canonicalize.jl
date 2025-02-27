function TensorKit.leftorth(elm::MPSTensor{3})
    return leftorth(elm.A,(1,2),(3,))
end

function TensorKit.leftorth(A::MPSTensor{R}) where R
    @assert R > 3
    Q,Rm = leftorth(A.A,(1,2),tuple(3:R...))
    return map(MPSTensor,(Q,permute(Rm,(1,),tuple(2:R-1...))))
end

function TensorKit.leftorth(A::MPSTensor{3}, B::MPSTensor{3})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3] ≔ Rm[-1,1]*B.A[1,-2,-3]
    return map(MPSTensor,[Q,tmp])
end

function TensorKit.leftorth!(obj::DenseMPS,site::Int64)
    obj.ts[site:site+1] = leftorth(obj.ts[site:site+1]...)
end

function TensorKit.rightorth(A::MPSTensor{3})
    Lm,Q = rightorth(A.A,(1,),(2,3))
    return Lm,permute(Q,(1,2),(3,))
end

function TensorKit.rightorth(A::MPSTensor{R}) where R
    @assert R > 3
    Lm,Q = rightorth(A.A,(1,2),tuple(3:R...))
    return map(MPSTensor,(Lm, permute(Q,(1,),tuple(2:R-1...))))
end

function TensorKit.rightorth(A::MPSTensor{3}, B::MPSTensor{3})
    Lm,Q = rightorth(B)
    return map(MPSTensor,[A.A*Lm,Q])
end

function TensorKit.rightorth!(obj::DenseMPS,site::Int64)
    obj.ts[site-1:site] = rightorth(obj.ts[site-1:site]...)
end



function TensorKit.leftorth(elm::DenseMPOTensor{4})
    return leftorth(elm.A,(1,2,4),(3,))
end

function TensorKit.leftorth!(A::DenseMPOTensor{4}, B::DenseMPOTensor{4})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3 -4] ≔ Rm[-2,1]*B.A[-1,1,-3,-4]
    A.A = permute(Q,(1,2),(4,3))
    B.A = tmp
    #return map(DenseMPOTensor,[Q,tmp])
end

function TensorKit.leftorth!(obj::DenseMPO,site::Int64)
    #obj.ts[site:site+1] = leftorth(obj.ts[site:site+1]...)
    leftorth!(obj.ts[site:site+1]...)
end

function TensorKit.rightorth(A::DenseMPOTensor{4})
    return rightorth(A.A,(2,),(1,3,4))
end

function TensorKit.rightorth!(A::DenseMPOTensor{4}, B::DenseMPOTensor{4})
    Lm,Q = rightorth(B)
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[-1,-2,1,-4]*Lm[1,-3]
    A.A = tmp
    B.A = permute(Q,(2,1),(3,4))
    #return map(DenseMPOTensor,[,])
end

function TensorKit.rightorth!(obj::DenseMPO,site::Int64)
    #obj.ts[site-1:site] = rightorth(obj.ts[site-1:site]...)
    rightorth!(obj.ts[site-1:site]...)
end


function TensorKit.tsvd(A::CompositeMPSTensor{2, R}; direction::Symbol=:center, kwargs...) where {R}
    @assert direction in [:center,:left,:right]
    U,S,V,ϵ = tsvd(A.A,(1,2),tuple(3:R...);kwargs...)
    ϵ /= sqrt(@tensor S[1,2] * S'[2,1])
    if direction == :center
        return U,S,V,ϵ
    elseif direction == :left 
        return map(MPSTensor,(U*S,permute(V,(1,2),tuple(3:(R-1)...))))...,ϵ
    elseif direction == :right 
        return map(MPSTensor,(U,permute(S*V,(1,2),tuple(3:(R-1)...))))...,ϵ
    end
end

function TensorKit.tsvd(A::CompositeMPOTensor{2,6}; direction::Symbol=:center, kwargs...)
    @assert direction in [:center,:left,:right]
    U,S,V,ϵ = tsvd(A.A,(2,3,6),(1,4,5);kwargs...)
    ϵ /= sqrt(@tensor S[1,2] * S'[2,1])
    if direction == :center
        return permute(U,(1,2),(4,3)),S,permute(V,(2,1),(3,4)),ϵ
    elseif direction == :left 
        return map(DenseMPOTensor,(permute(U*S,(1,2),(4,3)),permute(V,(2,1),(3,4))))...,ϵ
    elseif direction == :right 
        return map(DenseMPOTensor,(permute(U,(1,2),(4,3)),permute(S*V,(2,1),(3,4))))...,ϵ
    end
end

function TensorKit.tsvd(A::MPSTensor{3}; direction::Symbol=:center, kwargs...)
    @assert direction in [:center,:left,:right]
    if direction == :center
        U,S,V,ϵ = tsvd(A.A,(1,2),(3,);kwargs...)
        ϵ /= sqrt(@tensor S[1,2] * S'[2,1])
        return U,S,V,ϵ
    elseif direction == :left 
        U,S,V,ϵ = tsvd(A.A,(1,),(2,3);kwargs...)
        ϵ /= sqrt(@tensor S[1,2] * S'[2,1])
        return map(MPSTensor,(U*S,permute(V,(1,2),(3,))))...,ϵ
    elseif direction == :right 
        U,S,V,ϵ = tsvd(A.A,(1,2),(3,);kwargs...)
        ϵ /= sqrt(@tensor S[1,2] * S'[2,1])
        return map(MPSTensor,(U,S*V))...,ϵ
    end
end

function canonicalize!(obj::Union{DenseMPO{L},DenseMPS{L}},sl::Int64,sr::Int64) where {L}
    @assert 1 ≤ sl ≤ sr ≤ L 

    for sli in obj.center[1]:sl-1
        leftorth!(obj,sli)
        obj.center[1] += 1
        ( obj.center[1] > obj.center[2] ) && ( obj.center[2] += 1 )
    end
    for sri in obj.center[2]:-1:sr+1
        rightorth!(obj,sri)
        obj.center[2] -= 1
        ( obj.center[1] > obj.center[2] ) && ( obj.center[1] -= 1 )
    end
    return obj
end

function canonicalize!(::SparseMPO, ::Int64) end

function canonicalize!(obj::Union{DenseMPO{L},DenseMPS{L}},si::Int64) where {L}
    @assert 1 ≤ si ≤ L 
    return canonicalize!(obj,si,si)
end

function normalize!(obj::Union{DenseMPO{L},DenseMPS{L}}) where {L}
    @assert 1 == obj.center[1] == obj.center[2]
    return normalize!(obj.ts[1])
end

function normalize!(obj::Union{AbstractMPOTensor,AbstractMPSTensor})
    tmp = norm(obj.A)
    obj.A = obj.A / tmp
    return tmp
end

function normalize!(obj::Union{DenseMPOTensor,MPSTensor})
    tmp = norm(obj.A)
    obj.A = obj.A / tmp
    return tmp
end

function orthogonalize!(env::Environment{3},B::MPSTensor{3},EnvR::SparseRightEnvironmentTensor,osite::Int64)
    w,w2 = env.layer[2].D[osite]
    EnvRorth = Vector(undef,w)
    EnvRorth .= nothing

    for i in 1:w, j in 1:w2
        Hij = env.layer[2].ts[osite].m[i,j]
        isnothing(Hij) && continue
        tmp = contract(B,Hij,EnvR.A[j])
        if isnothing(EnvRorth[i])
            EnvRorth[i] = tmp - contract(tmp,B)
        else
            EnvRorth[i] += tmp - contract(tmp,B)
        end
    end

    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
end

function orthogonalize!(env::Environment{3},A::MPSTensor{3},EnvL::SparseLeftEnvironmentTensor,osite::Int64)
    w1,w = env.layer[2].D[osite]
    EnvLorth = Vector(undef,w)
    EnvLorth .= nothing
    @assert EnvL.D == w1

    for i in 1:w1, j in 1:w
        Hij = env.layer[2].ts[osite].m[i,j]
        isnothing(Hij) && continue
        tmp = contract(EnvL.A[i],A,Hij)
        if isnothing(EnvLorth[j])
            EnvLorth[j] = tmp - contract(tmp,A)
        else
            EnvLorth[j] += tmp - contract(tmp,A)
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
end

#= function normalize!(obj::SparseCompositeMPOTensor{N,R}) where {N,R}
    @assert (center = obj.center[1]) == obj.center[2]
    tmp = norm(obj.A)
    obj.A[center] /= tmp
    return tmp
end =#
