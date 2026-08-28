
function tsvd(A::CompositeMPSTensor{2, R}; direction::Symbol=:center, kwargs...) where {R}
    @assert direction in [:center,:left,:right]
    trunc = get(kwargs,:trunc,TruncationScheme())[:]
    U,S,V,ϵ = tsvd(A.A,(1,2),tuple(3:R...);trunc = trunc)
    if direction == :center
        return map(MPSTensor,[U,S,permute(V, ((1, 2), (3,)))])...,ϵ^2,BondInfo(S)
    elseif direction == :left 
        return map(MPSTensor,(U*S,permute(V, ((1, 2), tuple(3:(R-1)...)))))...,ϵ^2,BondInfo(S)
    elseif direction == :right
        return map(MPSTensor,(U,permute(S*V, ((1, 2), tuple(3:(R-1)...)))))...,ϵ^2,BondInfo(S)
    end
end

function tsvd(A::MPSTensor{3}; direction::Symbol=:center, index_tuple = ((1,2),(3,)), kwargs...)
    @assert direction in [:center,:left,:right]
    trunc = get(kwargs,:trunc,TruncationScheme())[:]
    if direction == :center
        U,S,V,ϵ = tsvd(A.A,index_tuple...;trunc = trunc)
        return map(MPSTensor,(U,S,V))...,ϵ^2,BondInfo(S)
    elseif direction == :left 
        U,S,V,ϵ = tsvd(A.A,(1,),(2,3);trunc = trunc)
        return map(MPSTensor,(U*S,permute(V, ((1, 2), (3,)))))...,ϵ^2,BondInfo(S)
    elseif direction == :right 
        U,S,V,ϵ = tsvd(A.A,(1,2),(3,);trunc = trunc)
        return map(MPSTensor,(U,S*V))...,ϵ^2,BondInfo(S)
    end
end

leftorth(A::MPSTensor{4}) = map(MPSTensor,leftorth(A.A,(1,2,4),(3,)) |> x -> (permute(x[1], ((1, 2), (4,3))),x[2]))
rightorth(A::MPSTensor{4}) = map(MPSTensor,rightorth(A.A,(1,),(2,3,4)) |> x -> (x[1],permute(x[2], ((1, 2), (3,4))))) 

rightorth(A::MPSTensor{3}) = map(MPSTensor,rightorth(A.A,(1,),(2,3)) |> x -> (x[1],permute(x[2], ((1, 2), (3,))))) 
leftorth(A::MPSTensor{3}) = map(MPSTensor,leftorth(A.A,(1,2),(3,)))

function rightorth(A::MPSTensor{3}, B::MPSTensor{3})
    Lm,Q = rightorth(B)
    return MPSTensor(A.A*Lm.A),Q
end

function leftorth(A::MPSTensor{3}, B::MPSTensor{3})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3] ≔ Rm.A[-1,1]*B.A[1,-2,-3]
    return Q,MPSTensor(tmp)
end

function rightorth!(obj::T,site::Int64) where T <: Union{DenseMPS, AdjointMPS}
    obj[site-1:site] = collect(rightorth(obj[site-1:site]...))
end

function leftorth!(obj::T,site::Int64) where T <: Union{DenseMPS, AdjointMPS}
    obj[site:site+1] = collect(leftorth(obj[site:site+1]...))
end


rightorth(A::AdjointMPSTensor{3}) = map(AdjointMPSTensor,leftorth(A.A,(1,3),(2,)) |> x -> (x[2],permute(x[1],((1,),(3,2))))) 
leftorth(A::AdjointMPSTensor{3}) = map(AdjointMPSTensor,rightorth(A.A,(1,),(2,3)) |> x -> (x[2],x[1]))

function rightorth(A::AdjointMPSTensor{3}, B::AdjointMPSTensor{3})
    Lm,Q = rightorth(B)
    return AdjointMPSTensor(Lm.A * A.A),Q
end

function leftorth(A::AdjointMPSTensor{3}, B::AdjointMPSTensor{3})
    Q, Rm = leftorth(A)
    @tensor tmp[-1;-2 -3] ≔ Rm.A[1,-2]*B.A[-1,1,-3]
    return Q,AdjointMPSTensor(tmp)
end

function leftorth(A::CompositeMPSTensor{2, 4})
    Q,R = leftorth(A.A,(1,2),(3,4))
    return map(MPSTensor, (permute(Q,((1,2),(3,))),permute(R,((1,2),(3,)))))
end

function rightorth(A::CompositeMPSTensor{2, 4})
    L,Q = rightorth(A.A,(1,2),(3,4))
    return map(MPSTensor, (permute(L,((1,2),(3,))),permute(Q,((1,2),(3,)))))
end
