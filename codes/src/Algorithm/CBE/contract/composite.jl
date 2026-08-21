

# composite
# MPS Sparse
contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, ::IdentityOperator{1}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3] ≔ El.A[-1,1] * A.A[1,-2,-3])
contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::LocalOperator{1,1}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3] ≔ El.A[-1,1] * A.A[1,2,-3] * mpo.A[-2,2])
contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::LocalOperator{1,2}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[1,2,-4] * mpo.A[-2,-3,2])
contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, ::IdentityOperator{1}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,-3,1] * A.A[1,-2,-4])
contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::LocalOperator{2,1}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3] ≔ El.A[-1,3,1] * A.A[1,2,-3] * mpo.A[-2,3,2])
contract(El::LeftEnvironmentTensor{3}, A::MPSTensor{3}, mpo::LocalOperator{1,1}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,-3,1] * A.A[1,2,-4] * mpo.A[-2,2])
contract(A::MPSTensor{3}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,2,1] * B.A[-3,2] * Er.A[1,-2,-4])
contract(A::MPSTensor{3}, B::LocalOperator{1,1}, EnvR::RightEnvironmentTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3] ≔ A.A[-1,2,1] * B.A[-2,2] * EnvR.A[1,-3])
contract(A::MPSTensor{3}, ::IdentityOperator{1}, EnvR::RightEnvironmentTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3] ≔ A.A[-1,-2,1] * EnvR.A[1,-3])
contract(A::MPSTensor{3}, B::LocalOperator{1,2}, EnvR::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3] ≔ A.A[-1,3,1] * B.A[-2,2,3] * EnvR.A[1,2,-3])
contract(A::MPSTensor{3}, ::IdentityOperator{1}, EnvR::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,-3,1] * EnvR.A[1,-2,-4])
contract(A::MPSTensor{3}, B::LocalOperator{2,1}, EnvR::RightEnvironmentTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,2,1] * B.A[-3,-2,2] * EnvR.A[1,-4])
# MPO Sparse
contract(El::LeftEnvironmentTensor{2},A::DenseMPOTensor{4}, B::LocalOperator{1,1}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[2,1,-3,-4] * B.A[-2,2])
contract(El::LeftEnvironmentTensor{2},A::DenseMPOTensor{4}, ::IdentityOperator{1}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[-2,1,-3,-4])
contract(El::LeftEnvironmentTensor{2},A::DenseMPOTensor{4}, B::LocalOperator{1,2}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,1] * A.A[2,1,-4,-5] * B.A[-2,-3,2])
contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::LocalOperator{2,1}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-1,2,1] * A.A[3,1,-3,-4] * B.A[-2,2,3])
contract(A::DenseMPOTensor{4}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,2] * Er.A[1,-2,-4])
contract(A::DenseMPOTensor{4}, ::IdentityOperator{1},Er::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[-3,-1,1,-5] * Er.A[1,-2,-4])
contract(A::DenseMPOTensor{4}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ A.A[2,-1,1,-4] * B.A[-2,2] * Er.A[1,-3])
contract(A::DenseMPOTensor{4}, ::IdentityOperator{1},Er::RightEnvironmentTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ A.A[-2,-1,1,-4] * Er.A[1,-3])
contract(A::DenseMPOTensor{4}, B::LocalOperator{2,1},Er::RightEnvironmentTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,-2,2] * Er.A[1,-4])
contract(A::DenseMPOTensor{4}, B::LocalOperator{1,2},Er::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ A.A[3,-1,1,-4] * B.A[-2,2,3] * Er.A[1,2,-3])
# MPO Dense
contract(El::LeftEnvironmentTensor{3},A::DenseMPOTensor{4}, B::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,2,1] * A.A[3,1,-4,-5] * B.A[-2,2,-3,3])
contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{4},Er::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[3,-1,1,-5] * B.A[-3,-2,2,3] * Er.A[1,2,-4])

contract(El::LeftEnvironmentTensor{3},A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,2,1] * A.A[3,1,-4,-5] * B.A[-3,-2,3,2])
contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, Er::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[3,-1,1,-5] * B.A[2,-3,3,-2] * Er.A[1,2,-4])
# MPS Dense
contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,3,1] * A.A[1,2,-4] * mpo.A[-2,3,-3,2])
contract(A::MPSTensor{3}, B::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{3}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,3,1] * B.A[-3,-2,2,3] * EnvR.A[1,2,-4])
# MPO Dense layer2
# contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-1,1] * A.A[-2,1,-3,-4])
# contract(EnvR::RightEnvironmentTensor{2}, A::DenseMPOTensor{4}) = RightCompositeEnvironmentTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvR.A[1,-3] * A.A[-2,-1,1,-4])
