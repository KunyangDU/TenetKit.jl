contract(El::LeftEnvironmentTensor{3},A::DenseMPOTensor{4}, B::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,2,1] * A.A[3,1,-4,-5] * B.A[-2,2,-3,3])
contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{4},Er::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[3,-1,1,-5] * B.A[-3,-2,2,3] * Er.A[1,2,-4])

contract(El::LeftEnvironmentTensor{3},A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,2,1] * A.A[3,1,-4,-5] * B.A[-3,-2,3,2])
contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, Er::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[3,-1,1,-5] * B.A[2,-3,3,-2] * Er.A[1,2,-4])

# contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-1,1] * A.A[-2,1,-3,-4])
# contract(EnvR::RightEnvironmentTensor{2}, A::DenseMPOTensor{4}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvR.A[1,-3] * A.A[-2,-1,1,-4])
