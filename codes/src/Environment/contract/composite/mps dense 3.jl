contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,3,1] * A.A[1,2,-4] * mpo.A[-2,3,-3,2])
contract(A::MPSTensor{3}, B::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,3,1] * B.A[-3,-2,2,3] * EnvR.A[1,2,-4])
