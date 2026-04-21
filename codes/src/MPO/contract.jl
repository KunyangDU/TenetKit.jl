# #= PUSH ENVIRONMENT =#
# """
# MPO + sparse MPO + ENVR
# push left
# """
# function contract(A::DenseMPOTensor{<:Number, 4}, B::SparseMPOTensor{N, M}, C::AdjointMPOTensor{<:Number, 4}, EnvR::SparseRightEnvironmentTensor) where {N,M}
#     @assert EnvR.D == M
#     tmpEnvR = Vector{Any}(nothing,N)
#     for i in 1:N, j in 1:M
#         isnothing(B.m[i,j]) && continue
#         if isnothing(tmpEnvR[i])
#             tmpEnvR[i] = contract(A, B.m[i,j], C, EnvR.A[j])
#         else 
#             tmpEnvR[i] += contract(A, B.m[i,j], C, EnvR.A[j])
#         end
#     end
#     return convert(Vector{RightEnvironmentTensor},tmpEnvR)
# end

# function contract(A::DenseMPOTensor{<:Number, 4}, B::SparseMPOTensor{N, M}, C::AdjointMPOTensor{<:Number, 4}, EnvL::SparseLeftEnvironmentTensor) where {N,M}
#     @assert EnvL.D == N
#     tmpEnvL = Vector{Any}(nothing,M)
#     for i in 1:N, j in 1:M
#         isnothing(B.m[i,j]) && continue
#         if isnothing(tmpEnvL[j])
#             tmpEnvL[j] = contract(EnvL.A[i], A, B.m[i,j], C)
#         else 
#             tmpEnvL[j] += contract(EnvL.A[i], A, B.m[i,j], C)
#         end
#     end
#     return convert(Vector{LeftEnvironmentTensor},tmpEnvL)
# end
# """
# ENVL + MPO + sparse MPO
# push right
# """
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,1] * A.A[2,1,-3,5] * B.A[4,-2,2] * C.A[-1,5,4,3]
#     return LeftEnvironmentTensor(tmp)
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1;-2] ≔ EnvL.A[4,2,1] * A.A[3,1,-2,6] * B.A[5,2,3] * C.A[-1,6,5,4]
#     return LeftEnvironmentTensor(tmp)
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1;-2] ≔ EnvL.A[3,1] * A.A[2,1,-2,5] * B.A[4,2] * C.A[-1,5,4,3]
#     return LeftEnvironmentTensor(tmp)
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1}, C::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1;-2] ≔ EnvL.A[3,1] * A.A[2,1,-2,4] * C.A[-1,4,2,3]
#     return LeftEnvironmentTensor(tmp)
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1}, C::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,-2,1] * A.A[2,1,-3,4] * C.A[-1,4,2,3]
#     return LeftEnvironmentTensor(tmp)
# end
# """
# MPOs + ENVR
# push left
# """
# function contract(A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1}, C::AdjointMPOTensor{<:Number, 4},EnvR::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1;-2] ≔ A.A[2,-1,1,4] * C.A[3,4,2,-2] * EnvR.A[1,3]
#     return RightEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{<:Number, 4},EnvR::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1;-2] ≔ A.A[2,-1,1,5] * B.A[4,2] * C.A[3,5,4,-2] * EnvR.A[1,3]
#     return RightEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{<:Number, 4},EnvR::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,5] * B.A[4,-2,2] * C.A[3,5,4,-3] * EnvR.A[1,3]
#     return RightEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{<:Number, 4},EnvR::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1;-2] ≔ A.A[3,-1,1,6] * B.A[5,2,3] * C.A[4,6,5,-2] * EnvR.A[1,2,4]
#     return RightEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1}, C::AdjointMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,4] * C.A[3,4,2,-3] * EnvR.A[1,-2,3]
#     return RightEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,4] * B.A[5,2] * C.A[3,4,5,-3] * EnvR.A[1,-2,3]
#     return RightEnvironmentTensor(tmp)
# end
# #= COMPOSITE ENVIRONMENT =#
# """
# CBE
# """
# function contract(El::LeftEnvironmentTensor{<:Number, 2},A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1})
#     @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[2,1,-3,-4] * B.A[-2,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{<:Number, 2},A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1})
#     @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[-2,1,-3,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{<:Number, 2},A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,2})
#     @tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,1] * A.A[2,1,-4,-5] * B.A[-2,-3,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{2,1})
#     @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-1,2,1] * A.A[3,1,-3,-4] * B.A[-2,2,3]
#     return LeftCompositeEnvironmentTensor(tmp)
# end


# function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,2] * Er.A[1,-2,-4]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1},Er::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[-3,-1,1,-5] * Er.A[1,-2,-4]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2;-3 -4] ≔ A.A[2,-1,1,-4] * B.A[-2,2] * Er.A[1,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1},Er::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2;-3 -4] ≔ A.A[-2,-1,1,-4] * Er.A[1,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{2,1},Er::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,-2,2] * Er.A[1,-4]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,2},Er::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2;-3 -4] ≔ A.A[3,-1,1,-4] * B.A[-2,2,3] * Er.A[1,2,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, EnvR::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,1,-4] * EnvR.A[1,-3]
#     return DenseMPOTensor(tmp)
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 5}, EnvR::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,2,1,-4] * EnvR.A[1,2,-3]
#     return DenseMPOTensor(tmp)
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2,4}, A::DenseMPOTensor{<:Number, 4})
#     LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-4])
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2,4}, B::DenseMPOTensor{<:Number, 4})
#     RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvR.A[-1,2,1,3] * B'.A[1,3,2,4] * B.A[-2,4,-3,-4])
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, Λ::DenseMPOTensor{<:Number, 2})
#     return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-1,-2,1,-4]*Λ.A[1,-3])
# end

# function contract(EnvL::RightCompositeEnvironmentTensor{<:Number, 2, 4}, Λ::DenseMPOTensor{<:Number, 2})
#     return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3,-4])
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, A::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1;-2] ≔ EnvL.A[1,2,-2,3] * A.A[-1,3,2,1] 
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 4}, A::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1;-2] ≔ EnvR.A[-1,2,1,3] * A.A[1,3,2,-2] 
#     return RightEnvironmentTensor(tmp)
# end

# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 4})
#     @tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,1] * EnvR.A[1,-1,-3,-4]
#     return DenseMPOTensor(tmp)
# end



# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2,5}, A::DenseMPOTensor{<:Number, 4})
#     LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[1,2,-3,-4,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-5])
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2,5}, B::DenseMPOTensor{<:Number, 4})
#     RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ EnvR.A[-1,-2,2,1,3] * B'.A[1,3,2,4] * B.A[-3,4,-4,-5])
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 5}, Λ::DenseMPOTensor{<:Number, 2})
#     return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[-1,-2,-3,1,-5]*Λ.A[1,-4])
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 5}, Λ::DenseMPOTensor{<:Number, 2})
#     return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ Λ.A[-1,1]*EnvR.A[1,-2,-3,-4,-5])
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 5}, A::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1;-2 -3] ≔ EnvL.A[1,2,-2,-3,3] * A.A[-1,3,2,1] 
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 5}, A::AdjointMPOTensor{<:Number, 4})
#     @tensor tmp[-1 -2;-3] ≔ EnvR.A[-1,-2,2,1,3] * A.A[1,3,2,-3] 
#     return RightEnvironmentTensor(tmp)
# end

# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 5})
#     @tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,2,1] * EnvR.A[1,2,-1,-3,-4]
#     return DenseMPOTensor(tmp)
# end

# """
# axpby!
# """
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,1] * A.A[-1,1,2,-4] * EnvR.A[2,-3]
#     return DenseMPOTensor(tmp)
# end

# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,1] * A.A[-2,1,2,-6] * B.A[-1,2,3,-5] * EnvR.A[3,-4]
#     return CompositeMPOTensor(tmp)
# end

# function contract(A::DenseMPOTensor{<:Number, 4}, B::AdjointMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1;-2] ≔ A.A[4,-1,1,3] * B.A[2,3,4,-2] * EnvR.A[1,2]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 4}, C::AdjointMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2;-3] ≔ A.A[3,-1,1,5] * B.A[6,-2,2,3] * C.A[4,5,6,-3] * EnvR.A[1,2,4]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::DenseMPOTensor{<:Number, 4}, B::AdjointMPOTensor{<:Number, 4}, EnvL::LeftEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1;-2] ≔ A.A[4,2,-2,3] * B.A[-1,3,4,1] * EnvL.A[1,2]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 4}, C::AdjointMPOTensor{<:Number, 4}, EnvL::LeftEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 ;-2 -3] ≔ A.A[6,4,-3,5] * B.A[3,2,-2,6] * C.A[-1,5,3,1] * EnvL.A[1,2,4]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(EnvL::DenseLeftEnvironmentTensor, A::DenseMPOTensor, B::DenseMPOTensor, EnvR::DenseRightEnvironmentTensor)
#     return contract(EnvL.A, A, B, EnvR.A)
# end
# function contract(EnvL::DenseLeftEnvironmentTensor, A::DenseMPOTensor, B::AdjointMPOTensor, EnvR::DenseRightEnvironmentTensor)
#     return contract(EnvL.A, A, B, EnvR.A)
# end

# """
# mul!
# """

# function contract(El::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, Er::RightCompositeEnvironmentTensor{<:Number, 2, 4})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ El.A[-3,-2,1,-6] * Er.A[1,-1,-4,-5]
#     return CompositeMPOTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{<:Number, 2, 5}, Er::RightCompositeEnvironmentTensor{<:Number, 2, 5})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ El.A[-3,-2,2,1,-6] * Er.A[1,2,-1,-4,-5]
#     return CompositeMPOTensor(tmp)
# end

# function contract(EnvL::SparseLeftEnvironmentTensor, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 4}, C::SparseMPOTensor{N₁,M₁}, D::SparseMPOTensor{N₂,M₂}, EnvR::SparseRightEnvironmentTensor) where {N₁,M₁,N₂,M₂}
#     @assert M₁ == N₂
#     tmp = nothing
#     for i in 1:N₁, j in 1:M₁, k in 1:M₂
#         isnothing(C.m[i,j]) | isnothing(D.m[j,k]) && continue
#         tmp1 = contract(EnvL.A[i], A, C.m[i,j])
#         tmp2 = contract(B, D.m[j,k], EnvR.A[k])
#         if isnothing(tmp)
#             tmp = contract(tmp1, tmp2)
#         else
#             tmp += contract(tmp1, tmp2)
#         end
#     end
#     return tmp
# end

# function contract(tr::DenseMPOTensor{<:Number, 2},obj::DenseMPOTensor{<:Number, 4})
#     return DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ tr.A[-2,1] * obj.A[-1,1,-3,-4])
# end

# function contract(obj::DenseMPOTensor{<:Number, 4},tl::DenseMPOTensor{<:Number, 2})
#     return DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ obj.A[-1,-2,1,-4] * tl.A[1,-3])
# end

# """
# make scalar
# """
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, B::AdjointMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 2})
#     return @tensor EnvL.A[2,1] * A.A[3,1,5,4] * B.A[6,4,3,2] * EnvR.A[5,6]
# end

# """
# tanTRG tools
# """

# function contract(B::AdjointMPOTensor{<:Number, 4}, A::DenseMPOTensor{<:Number, 4})
#     return @tensor A.A[3,1,2,4] * B.A[2,4,3,1]
# end

# function contract(B::AdjointCompositeMPOTensor{<:Number, 2,6}, A::CompositeMPOTensor{<:Number, 2,6})
#     return  @tensor A.A[5,6,2,1,3,4] * B.A[1,3,4,5,6,2]
# end

# """
# MPO + ENVR
# make composite ENVR (MPO + ENVR 1)
# """
# function contract(A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 2},Er::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2;-3 -4] ≔ A.A[2,-1,1,-4] * B.A[-2,2] * Er.A[1,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end


# function contract(A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 3},Er::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2;-3 -4] ≔ A.A[3,-1,1,-4] * B.A[-2,2,3] * Er.A[1,2,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end

# """
# ENVL + MPO
# make composite ENVL (ENVL + MPO 1)
# """
# # function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 3})
# #     @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-1,2,1] * A.A[3,1,-3,-4] * B.A[-2,2,3]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end
# """
# ENVL + composite MPO
# make composite ENVL (ENVL + composite MPO 1)
# """
# # function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::CompositeMPOTensor{<:Number, 2, 6}, B::DenseMPOTensor{<:Number, 3})
# #     @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,1] * A.A[-3,2,1,-5,-6,-7] * B.A[-2,-4,2]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::CompositeMPOTensor{<:Number, 2, 6}, B::LocalOperator{1,2})
#     @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,1] * A.A[-3,2,1,-5,-6,-7] * B.A[-2,-4,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# # function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::CompositeMPOTensor{<:Number, 2, 6}, B::DenseMPOTensor{<:Number, 3})
# #     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,2,1] * A.A[-3,3,1,-4,-5,-6] * B.A[-2,2,3]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::CompositeMPOTensor{<:Number, 2, 6}, B::LocalOperator{2,1})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,2,1] * A.A[-3,3,1,-4,-5,-6] * B.A[-2,2,3]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::CompositeMPOTensor{<:Number, 2, 6}, B::LocalOperator{1,1})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,1] * A.A[-3,2,1,-4,-5,-6] * B.A[-2,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::CompositeMPOTensor{<:Number, 2, 6}, ::IdentityOperator{1})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,1] * A.A[-3,-2,1,-4,-5,-6]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# # function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::CompositeMPOTensor{<:Number, 2, 6}, B::DenseMPOTensor{<:Number, 2})
# #     @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-4,1] * A.A[-3,2,1,-5,-6,-7] * B.A[-2,2]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::CompositeMPOTensor{<:Number, 2, 6}, B::LocalOperator{1,1})
#     @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-4,1] * A.A[-3,2,1,-5,-6,-7] * B.A[-2,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::CompositeMPOTensor{<:Number, 2, 6}, ::IdentityOperator{1})
#     @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-4,1] * A.A[-3,-2,1,-5,-6,-7]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# """
# MPO + ENVR
# make composite ENVR (MPO + ENVR)
# """
# # function contract(A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 2},Er::RightEnvironmentTensor{<:Number, 3})
# #     @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,2] * Er.A[1,-2,-4]
# #     return RightCompositeEnvironmentTensor(tmp)
# # end

# """
# ENVL + MPO
# make composite ENVL (ENVL + MPO 1)
# """
# # function contract(El::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 2})
# #     @tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,-3,1] * A.A[2,1,-4,-5] * B.A[-2,2]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end
# function contract(El::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1})
#     @tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,-3,1] * A.A[2,1,-4,-5] * B.A[-2,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1})
#     @tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,-3,1] * A.A[-2,1,-4,-5]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# """
# composite ENVL (ENVL + MPO 1) + MPO
# make composite ENVL (ENVL + MPO 1)
# """
# # function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 7}, A::DenseMPOTensor{<:Number, 3})
# #     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,-2,2,1,-4,-5,-6] * A.A[-3,1,2]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end
# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 7}, A::LocalOperator{2,1})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,-2,2,1,-4,-5,-6] * A.A[-3,1,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# # function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 6}, A::DenseMPOTensor{<:Number, 3})
# #     @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-2,1,-5,-6,-7] * A.A[-3,-4,1]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end
# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 6}, A::LocalOperator{1,2})
#     @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-2,1,-5,-6,-7] * A.A[-3,-4,1]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# # function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 6}, A::DenseMPOTensor{<:Number, 2})
# #     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,-2,1,-4,-5,-6] * A.A[-3,1]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end
# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 6}, A::LocalOperator{1,1})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,-2,1,-4,-5,-6] * A.A[-3,1]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 6}, ::IdentityOperator{1})
#     # @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,-2,-3,-4,-5,-6]
#     return EnvL
# end

# # function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 7}, A::DenseMPOTensor{<:Number, 2})
# #     @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-2,1,-4,-5,-6,-7] * A.A[-3,1]
# #     return LeftCompositeEnvironmentTensor(tmp)
# # end

# #= EFFECTIVE MPO =#

# """
# composite ENVL (ENVL + MPO 1) + composite ENVR (MPO + ENVR 1)
# make eff composite MPO
# """



# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 4})
#     @tensor tmp[-1 -2;-3 -4 -5] ≔ EnvL.A[-1,1,2] * A.A[-2,1,-3,3] * B.A[3,2,-4,-5]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2,5}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 4})
#     @tensor tmp[-1 -2;-3 -4 -5] ≔ EnvL.A[] * A.A[] * B.A[]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
"""
MPO + ENVR
make composite ENVR (MPO + ENVR 1)
"""


"""
ENVL + MPOs + ENVR
make eff MPOs
"""
# function contract(EnvL::DenseLeftEnvironmentTensor{3}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 4}, C::DenseMPOTensor{<:Number, 4}, D::DenseMPOTensor{<:Number, 4}, EnvR::DenseRightEnvironmentTensor{3})
#     # @tensor tmp1[-1 -2;-3 -4 -5] ≔ EnvL.A.A[-1,1,2] * A.A[-2,1,-3,3] * C.A[3,2,-4,-5]
#     # @tensor tmp2[-1 -2 -3;-4 -5] ≔ B.A[3,-1,1,-5] * D.A[-3,-2,2,3] * EnvR.A.A[1,2,-4]
#     # @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ tmp1[-3,-2,2,1,-6] * tmp2[1,2,-1,-4,-5]
#     # return CompositeMPOTensor(tmp)
#     return contract(EnvL.A,A,B,C,D,EnvR.A)
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 4}, C::DenseMPOTensor{<:Number, 4}, D::DenseMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 3})
#     @show "----------------testtesttest----------------"
#     @tensor tmp1[-1 -2;-3 -4 -5] ≔ EnvL.A[-1,1,2] * A.A[-2,1,-3,3] * C.A[3,2,-4,-5]
#     @tensor tmp2[-1 -2 -3;-4 -5] ≔ B.A[3,-1,1,-5] * D.A[-3,-2,2,3] * EnvR.A[1,2,-4]
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ tmp1[-3,-2,2,1,-6] * tmp2[1,2,-1,-4,-5]
#     return CompositeMPOTensor(tmp)
# end


"""
sparse ENVL + 2*MPO + 2*sparse MPO + sparse ENVR
make eff composite MPO
"""

# function contract(EnvL::SparseLeftEnvironmentTensor, A::SparseMPOTensor{N₁,M₁}, B::SparseMPOTensor{N₂,M₂}, C::AdjointMPOTensor{<:Number, 4}, D::AdjointMPOTensor{<:Number, 4}, EnvR::SparseRightEnvironmentTensor) where {N₁,M₁,N₂,M₂}
#     @assert M₁ == N₂
#     tmp = nothing
#     for i in 1:N₁, j in 1:M₁, k in 1:M₂
#         isnothing(A.m[i,j]) | isnothing(B.m[j,k]) && continue
#         tmp1 = contract(EnvL.A[i], A.m[i,j], C)
#         tmp2 = contract(D, B.m[j,k], EnvR.A[k])
#         if isnothing(tmp)
#             tmp = contract(tmp1, tmp2)
#         else
#             tmp += contract(tmp1, tmp2)
#         end
#     end
#     return tmp
# end

# """
# composite ENVL (ENVL + MPO 1) + ENVR
# make eff MPO
# """


# """
# composite ENVL (ENVL + composite MPO 1) + ENVR
# make eff composite MPO
# """
# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 6}, EnvR::RightEnvironmentTensor{<:Number, 2})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,-2,-1,1,-5,-6] * EnvR.A[1,-4]
#     return CompositeMPOTensor(tmp)
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 3, 7}, EnvR::RightEnvironmentTensor{<:Number, 3})
#     @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,-2,-1,2,1,-5,-6] * EnvR.A[1,2,-4]
#     return CompositeMPOTensor(tmp)
# end
"""
ENVL + 2*element + 2*MPO + ENVR
make eff composite element
"""
# function contract(EnvL::LeftEnvironmentTensor, A::Union{MPSTensor,DenseMPOTensor}, B::Union{MPSTensor,DenseMPOTensor}, C::Union{MPSTensor,DenseMPOTensor}, D::Union{MPSTensor,DenseMPOTensor}, EnvR::RightEnvironmentTensor)
#     return contract(contract(EnvL, A, C), contract(B, D, EnvR))
# end

"""
sparse ENVL + MPOs + sparse ENVR
make eff MPOs
"""
# function contract(EnvL::SparseLeftEnvironmentTensor, A::DenseMPOTensor{<:Number, 4}, B::SparseMPOTensor{N,M}, C::Union{DenseMPOTensor{<:Number, 4},AdjointMPOTensor{<:Number, 4}}, EnvR::SparseRightEnvironmentTensor) where {N,M}
#     tmp = nothing
#     for i in 1:N, j in 1:M
#         isnothing(B.m[i,j]) && continue
#         tmp1 = contract(EnvL.A[i], A, B.m[i,j], C, EnvR.A[j])
#         if isnothing(tmp)
#             tmp = tmp1
#         else
#             tmp += tmp1
#         end
#     end
#     return tmp
# end

#= SCALAR =#



# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 2}, C::AdjointMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 2})
#     return @tensor EnvL.A[3,1] * A.A[2,1,6,5] * B.A[4,2] * C.A[7,5,4,3] * EnvR.A[6,7]
# end
"""
ENVL + MPO + ENVR
make scalar
"""
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, B::DenseMPOTensor{<:Number, 3}, C::AdjointMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 3})
#     return @tensor EnvL.A[3,1] * A.A[2,1,6,5] * B.A[4,7,2] * C.A[8,5,4,3] * EnvR.A[6,7,8]
# end
# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{<:Number, 4}, EnvR::RightEnvironmentTensor{<:Number, 3})
#     return @tensor EnvL.A[3,1] * A.A[2,1,6,5] * B.A[4,7,2] * C.A[8,5,4,3] * EnvR.A[6,7,8]
# end
#= INTRODUCTION =#



#= ================================================ =#

# function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4})
#     @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-1,1] * A.A[-2,1,-3,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(EnvR::RightEnvironmentTensor{<:Number, 2}, A::DenseMPOTensor{<:Number, 4})
#     @tensor tmp[-1 -2;-3 -4] ≔ A.A[-2,-1,1,-4] * EnvR.A[1,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end

# contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{<:Number, 4},EnvL::LeftEnvironmentTensor{<:Number, 2}) = contract(EnvL,A,B,C)
# contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{<:Number, 4},EnvL::LeftEnvironmentTensor{<:Number, 3}) = contract(EnvL,A,B,C)
# contract(A::DenseMPOTensor{<:Number, 4}, B::Union{LocalOperator{1,1}, IdentityOperator{1}}, C::AdjointMPOTensor{<:Number, 4},EnvL::LeftEnvironmentTensor{<:Number, 3}) = contract(EnvL,A,B,C)
# contract(A::DenseMPOTensor{<:Number, 4}, B::Union{LocalOperator{1,1}, IdentityOperator{1}}, C::AdjointMPOTensor{<:Number, 4},EnvL::LeftEnvironmentTensor{<:Number, 2}) = contract(EnvL,A,B,C)

