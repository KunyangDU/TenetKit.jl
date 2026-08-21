
# ================= MPS ================= #

function _action2_contract(obj::CompositeMPSTensor{2,4},El::LeftEnvironmentTensor{2},hl::LocalOperator{1,1},hr::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
    return @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,2,3,4] * hl.A[-2,2] * hr.A[-3,3] * Er.A[4,-4]
end

function _action2_contract(obj::CompositeMPSTensor{2,4},El::LeftEnvironmentTensor{2},::IdentityOperator{1},hr::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
    return @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,-2,3,4] * hr.A[-3,3] * Er.A[4,-4]
end

function _action2_contract(obj::CompositeMPSTensor{2,4},El::LeftEnvironmentTensor{2},hl::LocalOperator{1,1},::IdentityOperator{1},Er::RightEnvironmentTensor{2})
    return @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,2,-3,4] * hl.A[-2,2] * Er.A[4,-4]
end

function _action2_contract(obj::CompositeMPSTensor{2,4},El::LeftEnvironmentTensor{2},::IdentityOperator{1},::IdentityOperator{1},Er::RightEnvironmentTensor{2})
    return @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,-2,-3,4] * Er.A[4,-4]
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 2}, hr::LocalOperator{2, 1}, Er::RightEnvironmentTensor{2})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,2,3,5] * hl.A[-2,4,2] * hr.A[-3,4,3] * Er.A[5,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 2}, hr::LocalOperator{1, 1}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,5] * obj.A[5,2,4,1] * hl.A[-2,3,2] * hr.A[-3,4] * Er.A[1,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 2}, ::IdentityOperator{1}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,4] * obj.A[4,2,-3,1] * hl.A[-2,3,2] * Er.A[1,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 1}, hr::LocalOperator{1, 2}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,5] * obj.A[5,4,2,1] * hl.A[-2,4] * hr.A[-3,3,2] * Er.A[1,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, hr::LocalOperator{1, 2}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,4] * obj.A[4,-2,2,1] * hr.A[-3,3,2] * Er.A[1,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{3}, hl::LocalOperator{2, 1}, hr::LocalOperator{1, 1}, Er::RightEnvironmentTensor{2})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,3,1] * obj.A[1,2,4,5] * hl.A[-2,3,2] * hr.A[-3,4] * Er.A[5,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{3}, hl::LocalOperator{2, 1}, ::IdentityOperator{1}, Er::RightEnvironmentTensor{2})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,3,1] * obj.A[1,2,-3,4] * hl.A[-2,3,2] * Er.A[4,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{3}, hl::LocalOperator{1, 1}, hr::LocalOperator{1, 1}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,3,1] * obj.A[1,4,5,2] * hl.A[-2,4] * hr.A[-3,5] * Er.A[2,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, ::IdentityOperator{1}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,3,1] * obj.A[1,-2,-3,2] * Er.A[2,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{3}, hl::LocalOperator{1, 1}, hr::LocalOperator{2, 1}, Er::RightEnvironmentTensor{2})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,3,1] * obj.A[1,4,2,5] * hl.A[-2,4] * hr.A[-3,3,2] * Er.A[5,-4]
    return x
end

function _action2_contract(obj::CompositeMPSTensor{2, 4}, El::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, hr::LocalOperator{2, 1}, Er::RightEnvironmentTensor{2})
    @tensor x[-1,-2,-3;-4] ≔ El.A[-1,3,1] * obj.A[1,-2,2,4] * hr.A[-3,3,2] * Er.A[4,-4]
    return x
end

# ================= MPO ================= #

# function _action2_contract(obj::CompositeMPOTensor{2,6},El::LeftEnvironmentTensor{2},hl::LocalOperator{1,1},hr::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-3,3] * obj.A[2,1,3,4,-5,-6] * hl.A[-2,1] * hr.A[-1,2] * Er.A[4,-4]
# end
# function _action2_contract(obj::CompositeMPOTensor{2,6},El::LeftEnvironmentTensor{2},::IdentityOperator{1},hr::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-3,2] * obj.A[1,-2,2,3,-5,-6] * hr.A[-1,1] * Er.A[3,-4]
# end
# function _action2_contract(obj::CompositeMPOTensor{2,6},El::LeftEnvironmentTensor{2},hl::LocalOperator{1,1},::IdentityOperator{1},Er::RightEnvironmentTensor{2})
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-3,2] * obj.A[-1,1,2,3,-5,-6] * hl.A[-2,1] * Er.A[3,-4]
# end
# function _action2_contract(obj::CompositeMPOTensor{2,6},El::LeftEnvironmentTensor{2},::IdentityOperator{1},::IdentityOperator{1},Er::RightEnvironmentTensor{2})
#     @tensor tmp[-1,-2,-3;-4,-5,-6] ≔ El.A[-3,1] * obj.A[-1,-2,1,2,-5,-6] * Er.A[2,-4]
# end

function _action2_contract(obj::CompositeMPOTensor{2, 6},El::LeftEnvironmentTensor{2},hl::LocalOperator{1,1},hr::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
    return @tensor tmp[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,1] * obj.A[3,2,1,4,-5,-6] * hl.A[-2,2] * hr.A[-3,3] * Er.A[4,-4]
end

function _action2_contract(obj::CompositeMPOTensor{2, 6},El::LeftEnvironmentTensor{2},::IdentityOperator{1},hr::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
    return @tensor tmp[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,1] * obj.A[3,-2,1,4,-5,-6] * hr.A[-3,3] * Er.A[4,-4]
end

function _action2_contract(obj::CompositeMPOTensor{2, 6},El::LeftEnvironmentTensor{2},hl::LocalOperator{1,1},::IdentityOperator{1},Er::RightEnvironmentTensor{2})
    return @tensor tmp[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,1] * obj.A[-3,2,1,4,-5,-6] * hl.A[-2,2] * Er.A[4,-4]
end

function _action2_contract(obj::CompositeMPOTensor{2, 6},El::LeftEnvironmentTensor{2},::IdentityOperator{1},::IdentityOperator{1},Er::RightEnvironmentTensor{2})
    return @tensor tmp[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,1] * obj.A[-3,-2,1,4,-5,-6] * Er.A[4,-4]
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 2}, hr::LocalOperator{2, 1}, Er::RightEnvironmentTensor{2})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,1] * obj.A[3,2,1,5,-5,-6] * hl.A[-2,4,2] * hr.A[-3,4,3] * Er.A[5,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 2}, hr::LocalOperator{1, 1}, Er::RightEnvironmentTensor{3})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,5] * obj.A[4,2,5,1,-5,-6] * hl.A[-2,3,2] * hr.A[-3,4] * Er.A[1,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 2}, ::IdentityOperator{1}, Er::RightEnvironmentTensor{3})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,4] * obj.A[-3,2,4,1,-5,-6] * hl.A[-2,3,2] * Er.A[1,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 1}, hr::LocalOperator{1, 2}, Er::RightEnvironmentTensor{3})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,5] * obj.A[2,4,5,1,-5,-6] * hl.A[-2,4] * hr.A[-3,3,2] * Er.A[1,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, hr::LocalOperator{1, 2}, Er::RightEnvironmentTensor{3})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,4] * obj.A[2,-2,4,1,-5,-6] * hr.A[-3,3,2] * Er.A[1,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{3}, hl::LocalOperator{2, 1}, hr::LocalOperator{1, 1}, Er::RightEnvironmentTensor{2})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,3,1] * obj.A[4,2,1,5,-5,-6] * hl.A[-2,3,2] * hr.A[-3,4] * Er.A[5,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{3}, hl::LocalOperator{2, 1}, ::IdentityOperator{1}, Er::RightEnvironmentTensor{2})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,3,1] * obj.A[-3,2,1,4,-5,-6] * hl.A[-2,3,2] * Er.A[4,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{3}, hl::LocalOperator{1, 1}, hr::LocalOperator{1, 1}, Er::RightEnvironmentTensor{3})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,3,1] * obj.A[5,4,1,2,-5,-6] * hl.A[-2,4] * hr.A[-3,5] * Er.A[2,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, ::IdentityOperator{1}, Er::RightEnvironmentTensor{3})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,3,1] * obj.A[-3,-2,1,2,-5,-6] * Er.A[2,3,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{3}, hl::LocalOperator{1, 1}, hr::LocalOperator{2, 1}, Er::RightEnvironmentTensor{2})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,3,1] * obj.A[2,4,1,5,-5,-6] * hl.A[-2,4] * hr.A[-3,3,2] * Er.A[5,-4]
    return x
end

function _action2_contract(obj::CompositeMPOTensor{2, 6}, El::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, hr::LocalOperator{2, 1}, Er::RightEnvironmentTensor{2})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,3,1] * obj.A[2,-2,1,4,-5,-6] * hr.A[-3,3,2] * Er.A[4,-4]
    return x
end