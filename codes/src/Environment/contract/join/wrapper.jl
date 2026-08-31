
# {0} join 的 DDA（lm === nothing）入口：转发到上面左环境在前的二参方法。
# DDA 路径 CBEenvironment.lm 为 nothing，randSVD! 的 contract(env.Lorth, R_trunc, nothing) 落到元素级。
contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightEnvironmentTensor{2}, ::Nothing) = contract(EnvL,EnvR)
contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, EnvR::RightEnvironmentTensor{3}, ::Nothing) = contract(EnvL,EnvR)
contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{2, 4},::Nothing) = contract(EnvL,EnvR)
contract(El::LeftEnvironmentTensor{3}, Er::RightCompositeEnvironmentTensor{2, 5}, ::Nothing) = contract(El,Er)
contract(El::LeftCompositeEnvironmentTensor{2, 4}, Er::RightEnvironmentTensor{3}, ::Nothing) = contract(El,Er)
contract(El::LeftEnvironmentTensor{3}, Er::RightCompositeEnvironmentTensor{1, 4}, ::Nothing) = contract(El,Er)
