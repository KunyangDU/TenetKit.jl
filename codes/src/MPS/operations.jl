# function TensorKit.leftorth(elm::MPSTensor{3})
#     Q,Rm = leftorth(elm.A,(1,2),(3,))
#     return map(MPSTensor,(Q,Rm))
# end

# function TensorKit.leftorth(A::MPSTensor{R}) where R
#     @assert R > 3
#     Q,Rm = leftorth(A.A,(1,2),tuple(3:R...))
#     return Q,Rm
# end
TensorKit.leftorth(A::MPSTensor{3}) = map(MPSTensor,leftorth(A.A,(1,2),(3,)))
TensorKit.leftorth(A::MPSTensor{4}) = map(MPSTensor,leftorth(A.A,(1,2,4),(3,)) |> x -> (permute(x[1],(1,2),(4,3)),x[2]))


function TensorKit.leftorth(A::MPSTensor{3}, B::MPSTensor{3})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3] ≔ Rm.A[-1,1]*B.A[1,-2,-3]
    return Q,MPSTensor(tmp)
end

function TensorKit.leftorth!(obj::DenseMPS,site::Int64)
    obj.ts[site:site+1] = collect(leftorth(obj.ts[site:site+1]...))
end

# function TensorKit.rightorth(A::MPSTensor{3})
#     Lm,Q = rightorth(A.A,(1,),(2,3))
#     return map(MPSTensor,(Lm,permute(Q,(1,2),(3,))))
# end

# function TensorKit.rightorth(A::MPSTensor{R}) where R
#     @assert R > 3
#     Lm,Q = rightorth(A.A,(1,2),tuple(3:R...))
#     return Lm, Q
# end

TensorKit.rightorth(A::MPSTensor{3}) = map(MPSTensor,rightorth(A.A,(1,),(2,3)) |> x -> (x[1],permute(x[2],(1,2),(3,)))) 
TensorKit.rightorth(A::MPSTensor{4}) = map(MPSTensor,rightorth(A.A,(1,),(2,3,4)) |> x -> (x[1],permute(x[2],(1,2),(3,4)))) 

function TensorKit.rightorth(A::MPSTensor{3}, B::MPSTensor{3})
    Lm,Q = rightorth(B)
    return MPSTensor(A.A*Lm.A),Q
end

function TensorKit.rightorth!(obj::DenseMPS,site::Int64)
    obj.ts[site-1:site] = collect(rightorth(obj.ts[site-1:site]...))
end



function TensorKit.tsvd(A::CompositeMPSTensor{2, R}; direction::Symbol=:center, kwargs...) where {R}
    @assert direction in [:center,:left,:right]
    vns = nothing
    U,S,V,ϵ = tsvd(A.A,(1,2),tuple(3:R...);kwargs...)
    if direction == :center
        return map(MPSTensor,[U,S,permute(V,(1,2),(3,))])...,ϵ
    elseif direction == :left 
        d = sqrt(@tensor S[1,2] * S'[2,1])
        if d != 0
            vns = vonNeumann(S)
        end
        return map(MPSTensor,(U*S,permute(V,(1,2),tuple(3:(R-1)...))))...,ϵ,vns
    elseif direction == :right 
        d = sqrt(@tensor S[1,2] * S'[2,1])
        if d != 0
            vns = vonNeumann(S)
        end
        return map(MPSTensor,(U,permute(S*V,(1,2),tuple(3:(R-1)...))))...,ϵ,vns
    end
end

function TensorKit.tsvd(A::MPSTensor{3}; direction::Symbol=:center, index_tuple = ((1,2),(3,)), kwargs...)
    @assert direction in [:center,:left,:right]
    if direction == :center
        U,S,V,ϵ = tsvd(A.A,index_tuple...;kwargs...)
        return map(MPSTensor,(U,S,V))...,ϵ
    elseif direction == :left 
        U,S,V,ϵ = tsvd(A.A,(1,),(2,3);kwargs...)
        return map(MPSTensor,(U*S,permute(V,(1,2),(3,))))...,ϵ
    elseif direction == :right 
        U,S,V,ϵ = tsvd(A.A,(1,2),(3,);kwargs...)
        return map(MPSTensor,(U,S*V))...,ϵ
    end
end

