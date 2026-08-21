
# projection
contract(EnvL::LeftCompositeEnvironmentTensor{2,3}, A::MPSTensor{3}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3])
contract(EnvR::RightCompositeEnvironmentTensor{1,3}, B::MPSTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,2,1] * B'.A[1,3,2] * B.A[3,-2,-3])
contract(EnvL::LeftCompositeEnvironmentTensor{2,4}, A::MPSTensor{3}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,-4] * A'.A[3,1,2] * A.A[-1,-2,3])
contract(EnvR::RightCompositeEnvironmentTensor{1,4}, B::MPSTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4] ≔ EnvR.A[-1,-2,2,1] * B'.A[1,3,2] * B.A[3,-3,-4])

contract(EnvL::LeftCompositeEnvironmentTensor{2,4}, A::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-4])
contract(EnvR::RightCompositeEnvironmentTensor{2,4}, B::DenseMPOTensor{4}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvR.A[-1,2,1,3] * B'.A[1,3,2,4] * B.A[-2,4,-3,-4])
contract(EnvL::LeftCompositeEnvironmentTensor{2,5}, A::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[1,2,-3,-4,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-5])
contract(EnvR::RightCompositeEnvironmentTensor{2,5}, B::DenseMPOTensor{4}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ EnvR.A[-1,-2,2,1,3] * B'.A[1,3,2,4] * B.A[-3,4,-4,-5])
