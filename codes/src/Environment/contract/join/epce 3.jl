# —— 左环境在前（El=Left，Er=Right），与顶层 contract(SparseLeft, SparseRight) 语义一致
contract(EnvL::LeftEnvironmentTensor{3}, EnvR::RightCompositeEnvironmentTensor{1, 4}) = MPSTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,2,1] * EnvR.A[1,2,-2,-3])
contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{1, 3}) = MPSTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,1] * EnvR.A[1,-2,-3])
contract(EnvL::LeftEnvironmentTensor{3}, EnvR::RightCompositeEnvironmentTensor{2, 5}) = DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,2,1] * EnvR.A[1,2,-1,-3,-4])
contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{2, 4}) = DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,1] * EnvR.A[1,-1,-3,-4])
contract(El::LeftCompositeEnvironmentTensor{2,3},Er::RightEnvironmentTensor{2}) = MPSTensor(@tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,1] * Er.A[1,-3])
contract(El::LeftCompositeEnvironmentTensor{2,4},Er::RightEnvironmentTensor{3}) = MPSTensor(@tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,2,1] * Er.A[1,2,-3])
contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightEnvironmentTensor{2}) = DenseMPOTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,1,-4] * EnvR.A[1,-3])
contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, EnvR::RightEnvironmentTensor{3}) = DenseMPOTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,2,1,-4] * EnvR.A[1,2,-3])