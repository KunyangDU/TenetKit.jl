
function _action0_contract(obj::T,El::LeftEnvironmentTensor{2},Er::RightEnvironmentTensor{2}) where T <: Union{MPSTensor{2}, DenseMPOTensor{2}}
    @tensor x[-1;-2] ≔ El.A[-1,1] * obj.A[1,2] * Er.A[2,-2]
    return x
end

function _action0_contract(obj::T,El::LeftEnvironmentTensor{3},Er::RightEnvironmentTensor{3}) where T <: Union{MPSTensor{2}, DenseMPOTensor{2}}
    @tensor x[-1;-2] ≔ El.A[-1,3,1] * obj.A[1,2] * Er.A[2,3,-2]
    return x
end
