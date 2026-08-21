
# retain
contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, A::AdjointMPSTensor{3}) = LeftEnvironmentTensor(@tensor tmp[-1;-2] ≔ EnvL.A[1,2,-2] * A.A[-1,1,2] )
contract(EnvR::RightCompositeEnvironmentTensor{1, 3}, A::AdjointMPSTensor{3}) = RightEnvironmentTensor(@tensor tmp[-1;-2] ≔ EnvR.A[-1,2,1] * A.A[1,-2,2] )
contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, A::AdjointMPSTensor{3}) = LeftEnvironmentTensor(@tensor tmp[-1;-2 -3] ≔ EnvL.A[1,2,-2,-3] * A.A[-1,1,2] )
contract(EnvR::RightCompositeEnvironmentTensor{1, 4}, A::AdjointMPSTensor{3}) = RightEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,-2,2,1] * A.A[1,-3,2])

contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, A::AdjointMPOTensor{4}) = LeftEnvironmentTensor(@tensor tmp[-1;-2] ≔ EnvL.A[1,2,-2,3] * A.A[-1,3,2,1] )
contract(EnvR::RightCompositeEnvironmentTensor{2, 4}, A::AdjointMPOTensor{4}) = RightEnvironmentTensor(@tensor tmp[-1;-2] ≔ EnvR.A[-1,2,1,3] * A.A[1,3,2,-2] )
contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, A::AdjointMPOTensor{4}) = LeftEnvironmentTensor(@tensor tmp[-1;-2 -3] ≔ EnvL.A[1,2,-2,-3,3] * A.A[-1,3,2,1] )
contract(EnvR::RightCompositeEnvironmentTensor{2, 5}, A::AdjointMPOTensor{4}) = RightEnvironmentTensor(@tensor tmp[-1 -2;-3] ≔ EnvR.A[-1,-2,2,1,3] * A.A[1,3,2,-3] )

