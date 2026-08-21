
# ================= MPS ================= #

function _action1_contract(obj::MPSTensor{3},El::LeftEnvironmentTensor{2},h::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
    @tensor x[-1,-2;-3] ≔ El.A[-1,1] * obj.A[1,2,3] * h.A[-2,2] * Er.A[3,-3]
    return x
end

function _action1_contract(obj::MPSTensor{3},El::LeftEnvironmentTensor{3},h::LocalOperator{1,1},Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2;-3] ≔ El.A[-1,3,1] * obj.A[1,4,2] * h.A[-2,4] * Er.A[2,3,-3]
    return x
end

function _action1_contract(obj::MPSTensor{3},El::LeftEnvironmentTensor{2},h::LocalOperator{1,2},Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2;-3] ≔ El.A[-1,1] * obj.A[1,2,3] * h.A[-2,4,2] * Er.A[3,4,-3]
    return x
end

function _action1_contract(obj::MPSTensor{3},El::LeftEnvironmentTensor{2},::IdentityOperator{1},Er::RightEnvironmentTensor{2})
    @tensor x[-1,-2;-3] ≔ El.A[-1,1] * obj.A[1,-2,2] * Er.A[2,-3]
    return x
end

function _action1_contract(obj::MPSTensor{3},El::LeftEnvironmentTensor{3},::IdentityOperator{1},Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2;-3] ≔ El.A[-1,3,1] * obj.A[1,-2,2] * Er.A[2,3,-3]
    return x
end

function _action1_contract(obj::MPSTensor{3},El::LeftEnvironmentTensor{3},h::LocalOperator{2,1},Er::RightEnvironmentTensor{2})
    @tensor x[-1,-2;-3] ≔ El.A[-1,3,1] * obj.A[1,2,4] * h.A[-2,3,2] * Er.A[4,-3]
    return x
end

# ================= MPO ================= #

function _action1_contract(obj::DenseMPOTensor{4},El::LeftEnvironmentTensor{2},h::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
    @tensor x[-2,-1;-3,-4] ≔ El.A[-1,1] * obj.A[2,1,3,-4] * h.A[-2,2] * Er.A[3,-3]
    return x
end

function _action1_contract(obj::DenseMPOTensor{4},El::LeftEnvironmentTensor{3},h::LocalOperator{1,1},Er::RightEnvironmentTensor{3})
    @tensor x[-2,-1;-3,-4] ≔ El.A[-1,3,1] * obj.A[4,1,2,-4] * h.A[-2,4] * Er.A[2,3,-3]
    return x
end

function _action1_contract(obj::DenseMPOTensor{4},El::LeftEnvironmentTensor{2},h::LocalOperator{1,2},Er::RightEnvironmentTensor{3})
    @tensor x[-2,-1;-3,-4] ≔ El.A[-1,1] * obj.A[2,1,3,-4] * h.A[-2,4,2] * Er.A[3,4,-3]
    return x
end

function _action1_contract(obj::DenseMPOTensor{4},El::LeftEnvironmentTensor{2},::IdentityOperator{1},Er::RightEnvironmentTensor{2})
    @tensor x[-2,-1;-3,-4] ≔ El.A[-1,1] * obj.A[-2,1,2,-4] * Er.A[2,-3]
    return x
end

function _action1_contract(obj::DenseMPOTensor{4},El::LeftEnvironmentTensor{3},::IdentityOperator{1},Er::RightEnvironmentTensor{3})
    @tensor x[-2,-1;-3,-4] ≔ El.A[-1,3,1] * obj.A[-2,1,2,-4] * Er.A[2,3,-3]
    return x
end

function _action1_contract(obj::DenseMPOTensor{4},El::LeftEnvironmentTensor{3},h::LocalOperator{2,1},Er::RightEnvironmentTensor{2})
    @tensor x[-2,-1;-3,-4] ≔ El.A[-1,3,1] * obj.A[2,1,4,-4] * h.A[-2,3,2] * Er.A[4,-3]
    return x
end