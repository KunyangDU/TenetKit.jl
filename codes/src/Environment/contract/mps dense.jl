function contract(A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[-1,3,1] * A′.A[2,-2,3] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[2,3,-2] * A′.A[-1,1,3] * EnvL.A[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, mpot::DenseMPOTensor{4}, B::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1,-2;-3] ≔ A.A[-1,2,1] * mpot.A[5,-2,3,2] * B.A[4,-3,5] * EnvR.A[1,3,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{4},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2 -3] ≔ A.A[4,5,-3] * mpot.A[2,3,-2,5] * B.A[-1,1,2] * EnvL.A[1,3,4]
    return LeftEnvironmentTensor(tmp)
end