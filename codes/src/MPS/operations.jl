

function TensorKit.tsvd(A::CompositeMPSTensor{2, R}; direction::Symbol=:center, kwargs...) where {R}
    @assert direction in [:center,:left,:right]
    vns = nothing
    U,S,V,ϵ = tsvd(A.A,(1,2),tuple(3:R...);kwargs...)
    if direction == :center
        return map(MPSTensor,[U,S,permute(V,(1,2),(3,))])...,ϵ^2
    elseif direction == :left 
        d = sqrt(@tensor S[1,2] * S'[2,1])
        if d != 0
            vns = vonNeumann(S)
        end
        return map(MPSTensor,(U*S,permute(V,(1,2),tuple(3:(R-1)...))))...,ϵ^2,vns
    elseif direction == :right 
        d = sqrt(@tensor S[1,2] * S'[2,1])
        if d != 0
            vns = vonNeumann(S)
        end
        return map(MPSTensor,(U,permute(S*V,(1,2),tuple(3:(R-1)...))))...,ϵ^2,vns
    end
end

function TensorKit.tsvd(A::MPSTensor{3}; direction::Symbol=:center, index_tuple = ((1,2),(3,)), kwargs...)
    @assert direction in [:center,:left,:right]
    if direction == :center
        U,S,V,ϵ = tsvd(A.A,index_tuple...;kwargs...)
        return map(MPSTensor,(U,S,V))...,ϵ^2
    elseif direction == :left 
        U,S,V,ϵ = tsvd(A.A,(1,),(2,3);kwargs...)
        return map(MPSTensor,(U*S,permute(V,(1,2),(3,))))...,ϵ^2
    elseif direction == :right 
        U,S,V,ϵ = tsvd(A.A,(1,2),(3,);kwargs...)
        return map(MPSTensor,(U,S*V))...,ϵ^2
    end
end

TensorKit.leftorth(A::MPSTensor{4}) = map(MPSTensor,leftorth(A.A,(1,2,4),(3,)) |> x -> (permute(x[1],(1,2),(4,3)),x[2]))
TensorKit.rightorth(A::MPSTensor{4}) = map(MPSTensor,rightorth(A.A,(1,),(2,3,4)) |> x -> (x[1],permute(x[2],(1,2),(3,4)))) 

TensorKit.rightorth(A::MPSTensor{3}) = map(MPSTensor,rightorth(A.A,(1,),(2,3)) |> x -> (x[1],permute(x[2],(1,2),(3,)))) 
TensorKit.leftorth(A::MPSTensor{3}) = map(MPSTensor,leftorth(A.A,(1,2),(3,)))

function TensorKit.rightorth(A::MPSTensor{3}, B::MPSTensor{3})
    Lm,Q = rightorth(B)
    return MPSTensor(A.A*Lm.A),Q
end

function TensorKit.leftorth(A::MPSTensor{3}, B::MPSTensor{3})
    Q, Rm = leftorth(A)
    @tensor tmp[-1 -2;-3] ≔ Rm.A[-1,1]*B.A[1,-2,-3]
    return Q,MPSTensor(tmp)
end

function TensorKit.rightorth!(obj::T,site::Int64) where T <: Union{DenseMPS, AdjointMPS}
    obj[site-1:site] = collect(rightorth(obj[site-1:site]...))
end

function TensorKit.leftorth!(obj::T,site::Int64) where T <: Union{DenseMPS, AdjointMPS}
    obj[site:site+1] = collect(leftorth(obj[site:site+1]...))
end


TensorKit.rightorth(A::AdjointMPSTensor{3}) = map(AdjointMPSTensor,leftorth(A.A,(1,3),(2,)) |> x -> (x[2],permute(x[1],(1,),(3,2)))) 
TensorKit.leftorth(A::AdjointMPSTensor{3}) = map(AdjointMPSTensor,rightorth(A.A,(1,),(2,3)) |> x -> (x[2],x[1]))

function TensorKit.rightorth(A::AdjointMPSTensor{3}, B::AdjointMPSTensor{3})
    Lm,Q = rightorth(B)
    return AdjointMPSTensor(Lm.A * A.A),Q
end

function TensorKit.leftorth(A::AdjointMPSTensor{3}, B::AdjointMPSTensor{3})
    Q, Rm = leftorth(A)
    @tensor tmp[-1;-2 -3] ≔ Rm.A[1,-2]*B.A[-1,1,-3]
    return Q,AdjointMPSTensor(tmp)
end

