
# projection
contract(EnvL::LeftCompositeEnvironmentTensor{2,3}, A::MPSTensor{3}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3])
contract(EnvR::RightCompositeEnvironmentTensor{1,3}, B::MPSTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,2,1] * B'.A[1,3,2] * B.A[3,-2,-3])
contract(EnvL::LeftCompositeEnvironmentTensor{2,4}, A::MPSTensor{3}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,-4] * A'.A[3,1,2] * A.A[-1,-2,3])
contract(EnvR::RightCompositeEnvironmentTensor{1,4}, B::MPSTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4] ≔ EnvR.A[-1,-2,2,1] * B'.A[1,3,2] * B.A[3,-3,-4])

contract(EnvL::LeftCompositeEnvironmentTensor{2,4}, A::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-4])
contract(EnvR::RightCompositeEnvironmentTensor{2,4}, B::DenseMPOTensor{4}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvR.A[-1,2,1,3] * B'.A[1,3,2,4] * B.A[-2,4,-3,-4])
contract(EnvL::LeftCompositeEnvironmentTensor{2,5}, A::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[1,2,-3,-4,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-5])
contract(EnvR::RightCompositeEnvironmentTensor{2,5}, B::DenseMPOTensor{4}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ EnvR.A[-1,-2,2,1,3] * B'.A[1,3,2,4] * B.A[-3,4,-4,-5])

# 原地正交化投影：x -= x·A'·A，融合减法（@tensor -= ≡ mul!(x,A'·A,-1,1)），
# 不物化整张投影结果再 axpy，省一次结果分配 + 一次遍历。与上面 contract 的 8 个模式一一对应。
function _orth_sub!(x::LeftCompositeEnvironmentTensor{2,3}, A::MPSTensor{3})
    @tensor x.A[-1,-2;-3] -= x.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3]
    return x
end
function _orth_sub!(x::LeftCompositeEnvironmentTensor{2,4}, A::MPSTensor{3})
    @tensor x.A[-1,-2;-3,-4] -= x.A[1,2,-3,-4] * A'.A[3,1,2] * A.A[-1,-2,3]
    return x
end
function _orth_sub!(x::LeftCompositeEnvironmentTensor{2,4}, A::DenseMPOTensor{4})
    @tensor x.A[-1,-2;-3,-4] -= x.A[1,2,-3,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-4]
    return x
end
function _orth_sub!(x::LeftCompositeEnvironmentTensor{2,5}, A::DenseMPOTensor{4})
    @tensor x.A[-1,-2;-3,-4,-5] -= x.A[1,2,-3,-4,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-5]
    return x
end
function _orth_sub!(x::RightCompositeEnvironmentTensor{1,3}, A::MPSTensor{3})
    @tensor x.A[-1,-2;-3] -= x.A[-1,2,1] * A'.A[1,3,2] * A.A[3,-2,-3]
    return x 
end
function _orth_sub!(x::RightCompositeEnvironmentTensor{1,4}, A::MPSTensor{3})
    @tensor x.A[-1,-2,-3;-4] -= x.A[-1,-2,2,1] * A'.A[1,3,2] * A.A[3,-3,-4]
    return x
end
function _orth_sub!(x::RightCompositeEnvironmentTensor{2,4}, A::DenseMPOTensor{4})
    @tensor x.A[-1,-2;-3,-4] -= x.A[-1,2,1,3] * A'.A[1,3,2,4] * A.A[-2,4,-3,-4]
    return x
end
function _orth_sub!(x::RightCompositeEnvironmentTensor{2,5}, A::DenseMPOTensor{4})
    @tensor x.A[-1,-2,-3;-4,-5] -= x.A[-1,-2,2,1,3] * A'.A[1,3,2,4] * A.A[-3,4,-4,-5]
    return x
end
