function contract(EnvL::DenseLeftEnvironmentTensor, A::MPSTensor, B::AdjointMPSTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[1,2] * A.A[2,3,4] * A′.A[5,1,3] * EnvR.A[4,5]
end

function contract(EnvL::DenseLeftEnvironmentTensor, A::DenseMPOTensor, B::DenseMPOTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end

function contract(EnvL::DenseLeftEnvironmentTensor, A::DenseMPOTensor, B::AdjointMPOTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,1] * A.A[-1,1,2,-4] * EnvR.A[2,-3]
    return DenseMPOTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,1] * A.A[-2,1,2,-6] * B.A[-1,2,3,-5] * EnvR.A[3,-4]
    return CompositeMPOTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[2,1] * A.A[3,1,5,4] * B.A[6,4,3,2] * EnvR.A[5,6]
end