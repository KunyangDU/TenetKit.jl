contract(A::DenseMPOTensor{4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{4},EnvL::LeftEnvironmentTensor{2}) = contract(EnvL,A,B,C)
contract(A::DenseMPOTensor{4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{4},EnvL::LeftEnvironmentTensor{3}) = contract(EnvL,A,B,C)
contract(A::DenseMPOTensor{4}, B::Union{LocalOperator{1,1}, IdentityOperator{1}}, C::AdjointMPOTensor{4},EnvL::LeftEnvironmentTensor{3}) = contract(EnvL,A,B,C)
contract(A::DenseMPOTensor{4}, B::Union{LocalOperator{1,1}, IdentityOperator{1}}, C::AdjointMPOTensor{4},EnvL::LeftEnvironmentTensor{2}) = contract(EnvL,A,B,C)

# function contract(A::DenseMPOTensor{4}, B::SparseMPOTensor{DL, D, DR, T}, C::AdjointMPOTensor{4}, EnvR::SparseRightEnvironmentTensor) where {DL, D, DR, T}
#     @assert EnvR.D[1] == DR
#     tmpEnvR = Vector{Any}(nothing, D)
#     r_map = _validind1(B, R2L())
#     for (j, r_pairs) in enumerate(r_map)
#         isempty(r_pairs) && continue
#         weighted_env = sum(w * EnvR[b] for (b, w) in r_pairs)
#         tmpEnvR[j] = axpy!(1, contract(A, B[j], C, weighted_env), tmpEnvR[j])
#     end
#     return convert(Vector{RightEnvironmentTensor}, tmpEnvR)
# end

# function contract(A::DenseMPOTensor{4}, B::SparseMPOTensor{DL, D, DR, T}, C::AdjointMPOTensor{4}, EnvL::SparseLeftEnvironmentTensor) where {DL, D, DR, T}
#     @assert EnvL.D[1] == DL
#     tmpEnvL = Vector{Any}(nothing, D)
#     l_map = _validind1(B, L2R())
#     for (j, l_pairs) in enumerate(l_map)
#         isempty(l_pairs) && continue
#         weighted_env = sum(w * EnvL[b] for (b, w) in l_pairs)
#         tmpEnvL[j] = axpy!(1, contract(weighted_env, A, B[j], C), tmpEnvL[j])
#     end
#     return convert(Vector{LeftEnvironmentTensor}, tmpEnvL)
# end

function contract(A::DenseMPOTensor{4}, ::IdentityOperator{1}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[2,-1,1,4] * C.A[3,4,2,-2] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[2,-1,1,5] * B.A[4,2] * C.A[3,5,4,-2] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,5] * B.A[4,-2,2] * C.A[3,5,4,-3] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[3,-1,1,6] * B.A[5,2,3] * C.A[4,6,5,-2] * EnvR.A[1,2,4]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, ::IdentityOperator{1}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,4] * C.A[3,4,2,-3] * EnvR.A[1,-2,3]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,4] * B.A[5,2] * C.A[3,4,5,-3] * EnvR.A[1,-2,3]
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,1] * A.A[2,1,-3,5] * B.A[4,-2,2] * C.A[-1,5,4,3]
    return LeftEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[4,2,1] * A.A[3,1,-2,6] * B.A[5,2,3] * C.A[-1,6,5,4]
    return LeftEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[3,1] * A.A[2,1,-2,5] * B.A[4,2] * C.A[-1,5,4,3]
    return LeftEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, ::IdentityOperator{1}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[3,1] * A.A[2,1,-2,4] * C.A[-1,4,2,3]
    return LeftEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, ::IdentityOperator{1}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,-2,1] * A.A[2,1,-3,4] * C.A[-1,4,2,3]
    return LeftEnvironmentTensor(tmp)
end