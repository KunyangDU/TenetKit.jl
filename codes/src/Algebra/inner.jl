function TensorKit.inner(A::Union{DenseMPO{L₁},DenseMPS{L₁}},B::Union{AdjointMPO{L₂},AdjointMPS{L₂}}) where {L₁,L₂}
    @assert L₁ == L₂
    return let env = Environment([A,B])
        initialize!(env)
        _scalar(env)
    end
end

function TensorKit.inner(A::Union{DenseMPO{L₁},DenseMPS{L₁}},O::SparseMPO,B::Union{AdjointMPO{L₂},AdjointMPS{L₂}}) where {L₁,L₂}
    @assert L₁ == L₂
    return let env = Environment([A,O,B])
        initialize!(env)
        _scalar(env)
    end
end

TensorKit.inner(A::T) where T <: Union{DenseMPO,DenseMPS} = inner(A,A')
TensorKit.inner(A::T) where T <: Union{AdjointMPO,AdjointMPS} = inner(A',A)
TensorKit.inner(A::T,O::SparseMPO) where T <: Union{DenseMPO,DenseMPS} = inner(A,O,A')
TensorKit.inner(A::T,O::SparseMPO) where T <: Union{AdjointMPO,AdjointMPS} = inner(A',O,A)

inner1(A::DenseMPO) = A.center[1] == A.center[2] ? norm(A[A.center[1]]) : (@assert A.center[1] == A.center[2])
inner1(A::AdjointMPO) = inner1(A')'



