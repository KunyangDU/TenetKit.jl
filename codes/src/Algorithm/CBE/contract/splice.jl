# splice
contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, Λ::MPSTensor{2}) = LeftCompositeEnvironmentTensor(EnvL.A*Λ.A)
contract(EnvL::RightCompositeEnvironmentTensor{1, 3}, Λ::MPSTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3])
contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, Λ::MPSTensor{2}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-1,-2,-3,1]*Λ.A[1,-4])
contract(EnvL::RightCompositeEnvironmentTensor{1, 4}, Λ::MPSTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3,-4])

contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, Λ::DenseMPOTensor{2}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-1,-2,1,-4]*Λ.A[1,-3])
contract(EnvL::RightCompositeEnvironmentTensor{2, 4}, Λ::DenseMPOTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3,-4])
contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, Λ::DenseMPOTensor{2}) = LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[-1,-2,-3,1,-5]*Λ.A[1,-4])
contract(EnvR::RightCompositeEnvironmentTensor{2, 5}, Λ::DenseMPOTensor{2}) = RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ Λ.A[-1,1]*EnvR.A[1,-2,-3,-4,-5])

