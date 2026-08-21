function TensorKit.inner(A::Union{DenseMPO{L₁},DenseMPS{L₁}},B::Union{AdjointMPO{L₂},AdjointMPS{L₂}};kwargs...) where {L₁,L₂}
    @assert L₁ == L₂
    return let env = Environment([A,B])
        initialize!(env;kwargs...)
        _scalar(env)
    end
end

function TensorKit.inner(A::Union{DenseMPO{L₁},DenseMPS{L₁}},O::SparseMPO,B::Union{AdjointMPO{L₂},AdjointMPS{L₂}};kwargs...) where {L₁,L₂}
    @assert L₁ == L₂
    return let env = Environment([A,O,B])
        initialize!(env;kwargs...)
        _scalar(env)
    end
end

TensorKit.inner(A::T) where T <: Union{DenseMPO,DenseMPS} = inner(A,A')
TensorKit.inner(A::T) where T <: Union{AdjointMPO,AdjointMPS} = inner(A',A)
TensorKit.inner(A::T,O::SparseMPO) where T <: Union{DenseMPO,DenseMPS} = inner(A,O,A')
TensorKit.inner(A::T,O::SparseMPO) where T <: Union{AdjointMPO,AdjointMPS} = inner(A',O,A)

inner1(A::DenseMPO) = A.center[1] == A.center[2] ? norm(A[A.center[1]]) : (@assert A.center[1] == A.center[2])
inner1(A::AdjointMPO) = inner1(A')'


TensorKit.inner(A::MPSTensor{3}, B::AdjointMPSTensor{3}) = @tensor A.A[1,3,2] * B.A[2,1,3]
TensorKit.inner(A::AdjointMPSTensor{2},B::MPSTensor{2}) = @tensor A.A[1,2] * B.A[2,1]
TensorKit.inner(A::AdjointMPSTensor{3},B::MPSTensor{3}) = @tensor A.A[1,2,3] * B.A[2,3,1]
TensorKit.inner(B::AdjointCompositeMPSTensor{2, 4}, A::CompositeMPSTensor{2, 4}) = @tensor A.A[1,3,4,2] * B.A[2,1,3,4]

TensorKit.inner(B::AdjointMPOTensor{4}, A::DenseMPOTensor{4}) = @tensor A.A[3,1,2,4] * B.A[2,4,3,1]
TensorKit.inner(B::AdjointCompositeMPOTensor{2,6}, A::CompositeMPOTensor{2,6}) = @tensor A.A[5,6,2,1,3,4] * B.A[1,3,4,5,6,2]

