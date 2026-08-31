contract(EnvL::LeftEnvironmentTensor{2}, hl::LocalOperator{1, 1}, Al::AdjointMPOTensor{4}) = LeftCompositeEnvironmentTensor((@tensor tmp[-1 -2;-4 -5] ≔ EnvL.A[1,-5] * hl.A[2,-4] * Al.A[-1,-2,2,1]),3,1)
contract(EnvL::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, Al::AdjointMPOTensor{4}) = LeftCompositeEnvironmentTensor((@tensor tmp[-1 -2;-4 -5] ≔ EnvL.A[1,-5] * Al.A[-1,-2,-4,1]),3,1)
contract(hr::LocalOperator{1, 1}, Ar::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2}) = RightCompositeEnvironmentTensor((@tensor tmp[-1 -3;-4 -5] ≔ hr.A[2,-5] * Ar.A[1,-3,2,-4] * EnvR.A[-1,1]),3,1)
contract(::IdentityOperator{1}, Ar::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2}) = RightCompositeEnvironmentTensor((@tensor tmp[-1 -3;-4 -5] ≔ Ar.A[1,-3,-5,-4] * EnvR.A[-1,1]),3,1)

contract(El::LeftCompositeEnvironmentTensor{2, 4, 3, 1}, Er::RightCompositeEnvironmentTensor{2, 4, 3, 1}) = AdjointCompositeMPOTensor(@tensor tmp[-1 -2 -3;-4 -5 -6] ≔ El.A[1,-3,-5,-6] * Er.A[-1,-2,1,-4])
