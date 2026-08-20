function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * B.A[3,-2,2] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * mpot.A[4,2] * B.A[3,-2,4] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{2,1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,2,1] * mpot.A[4,-2,2] * B.A[3,-3,4] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ A.A[-1,2,1] * B.A[4,-3,2] * EnvR.A[1,-2,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,2},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * mpot.A[5,3,2] * B.A[4,-2,5] * EnvR.A[1,3,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[3,4,-2] * mpot.A[2,4] * B.A[-1,1,2] * EnvL.A[1,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[3,2,-2] * B.A[-1,1,2] * EnvL.A[1,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,2},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2 -3] ≔ A.A[3,4,-3] * mpot.A[2,-2,4] * B.A[-1,1,2] * EnvL.A[1,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2 -3] ≔ A.A[3,2,-3] * B.A[-1,1,2] * EnvL.A[1,-2,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{2,1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[4,5,-2] * mpot.A[3,2,5] * B.A[-1,1,3] * EnvL.A[1,2,4]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, mpot::LocalOperator{1, 1}, B::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1,-2;-3] ≔ A.A[-1,2,1] * mpot.A[4,2] * B.A[3,-3,4] * EnvR.A[1,-2,3]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2 -3] ≔ A.A[3,4,-3] * mpot.A[2,4] * B.A[-1,1,2] * EnvL.A[1,-2,3]
    return LeftEnvironmentTensor(tmp)
end
