
# #= PUSH ENVIRONMENT =#
# """
# MPS + ENVR
# push left 
# """
# function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * B.A[3,-2,2] * EnvR.A[1,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},mpot::LocalOperator{1,1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * mpot.A[4,2] * B.A[3,-2,4] * EnvR.A[1,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},mpot::LocalOperator{2,1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1 -2;-3] ≔ A.A[-1,2,1] * mpot.A[4,-2,2] * B.A[3,-3,4] * EnvR.A[1,3]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
#     @tensor tmp[-1 -2 ; -3] ≔ A.A[-1,2,1] * B.A[4,-3,2] * EnvR.A[1,-2,4]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},mpot::LocalOperator{1,2},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
#     @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * mpot.A[5,3,2] * B.A[4,-2,5] * EnvR.A[1,3,4]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},mpot::LocalOperator{1,1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[3,4,-2] * mpot.A[2,4] * B.A[-1,1,2] * EnvL.A[1,3]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[3,2,-2] * B.A[-1,1,2] * EnvL.A[1,3]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},mpot::LocalOperator{1,2},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
#     @tensor tmp[-1;-2 -3] ≔ A.A[3,4,-3] * mpot.A[2,-2,4] * B.A[-1,1,2] * EnvL.A[1,3]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
#     @tensor tmp[-1 -2 ; -3] ≔ A.A[3,2,-3] * B.A[-1,1,2] * EnvL.A[1,-2,3]
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3},mpot::LocalOperator{2,1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
#     @tensor tmp[-1;-2] ≔ A.A[4,5,-2] * mpot.A[3,2,5] * B.A[-1,1,3] * EnvL.A[1,2,4]
#     return LeftEnvironmentTensor(tmp)
# end


# """
# CBE
# """
# function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, ::IdentityOperator{1})
#     @tensor tmp[-1 -2;-3] ≔ El.A[-1,1] * A.A[1,-2,-3]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::LocalOperator{1,1})
#     @tensor tmp[-1 -2;-3] ≔ El.A[-1,1] * A.A[1,2,-3] * mpo.A[-2,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::LocalOperator{1,2})
#     @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[1,2,-4] * mpo.A[-2,-3,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, ::IdentityOperator{1})
#     @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,-3,1] * A.A[1,-2,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::LocalOperator{2,1})
#     @tensor tmp[-1 -2;-3] ≔ El.A[-1,3,1] * A.A[1,2,-3] * mpo.A[-2,3,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# function contract(A::MPSTensor{3}, B::LocalOperator{1,1}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1 -2;-3] ≔ A.A[-1,2,1] * B.A[-2,2] * EnvR.A[1,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::MPSTensor{3}, ::IdentityOperator{1}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1 -2;-3] ≔ A.A[-1,-2,1] * EnvR.A[1,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::MPSTensor{3}, B::LocalOperator{1,2}, EnvR::RightEnvironmentTensor{3})
#     @tensor tmp[-1 -2;-3] ≔ A.A[-1,3,1] * B.A[-2,2,3] * EnvR.A[1,2,-3]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::MPSTensor{3}, ::IdentityOperator{1}, EnvR::RightEnvironmentTensor{3})
#     @tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,-3,1] * EnvR.A[1,-2,-4]
#     return RightCompositeEnvironmentTensor(tmp)
# end
# function contract(A::MPSTensor{3}, B::LocalOperator{2,1}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,2,1] * B.A[-3,-2,2] * EnvR.A[1,-4]
#     return RightCompositeEnvironmentTensor(tmp)
# end

# function contract(EnvL::SparseLeftEnvironmentTensor,EnvR::SparseRightEnvironmentTensor)
#     @assert (w = EnvL.D) == EnvR.D
#     mps = nothing 
#     for i in 1:w 
#         tmp = contract(EnvL.A[i],EnvR.A[i])
#         mps = axpy!(1,tmp,mps)
#     end
#     return mps
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, A::AdjointMPSTensor{3})
#     @tensor tmp[-1;-2] ≔ EnvL.A[1,2,-2] * A.A[-1,1,2] 
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{1, 3}, A::AdjointMPSTensor{3})
#     @tensor tmp[-1;-2] ≔ EnvR.A[-1,2,1] * A.A[1,-2,2] 
#     return RightEnvironmentTensor(tmp)
# end

# function contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{1, 3})
#     @tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,1] * EnvR.A[1,-2,-3]
#     return MPSTensor(tmp)
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{2,3}, A::MPSTensor{3})
#     LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3])
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{1,3}, B::MPSTensor{3})
#     RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,2,1] * B'.A[1,3,2] * B.A[3,-2,-3])
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, Λ::MPSTensor{2})
#     return LeftCompositeEnvironmentTensor(EnvL.A*Λ.A)
# end

# function contract(EnvL::RightCompositeEnvironmentTensor{1, 3}, Λ::MPSTensor{2})
#     return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3])
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{2,4}, A::MPSTensor{3})
#     LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,-4] * A'.A[3,1,2] * A.A[-1,-2,3])
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{1,4}, B::MPSTensor{3})
#     RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4] ≔ EnvR.A[-1,-2,2,1] * B'.A[1,3,2] * B.A[3,-3,-4])
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, Λ::MPSTensor{2})
#     return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-1,-2,-3,1]*Λ.A[1,-4])
# end

# function contract(EnvL::RightCompositeEnvironmentTensor{1, 4}, Λ::MPSTensor{2})
#     return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3,-4])
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, A::AdjointMPSTensor{3})
#     @tensor tmp[-1;-2 -3] ≔ EnvL.A[1,2,-2,-3] * A.A[-1,1,2] 
#     return LeftEnvironmentTensor(tmp)
# end

# function contract(EnvR::RightCompositeEnvironmentTensor{1, 4}, A::AdjointMPSTensor{3})
#     @tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,-2,2,1] * A.A[1,-3,2] 
#     return RightEnvironmentTensor(tmp)
# end

# function contract(EnvL::LeftEnvironmentTensor{3}, EnvR::RightCompositeEnvironmentTensor{1, 4})
#     @tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,2,1] * EnvR.A[1,2,-2,-3]
#     return MPSTensor(tmp)
# end

# function contract(El::LeftCompositeEnvironmentTensor{2,3},Er::RightEnvironmentTensor{2})
#     @tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,1] * Er.A[1,-3]
#     return MPSTensor(tmp)
# end
# function contract(El::LeftCompositeEnvironmentTensor{2,4},Er::RightEnvironmentTensor{3})
#     @tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,2,1] * Er.A[1,2,-3]
#     return MPSTensor(tmp)
# end


# """
# DMRG tools
# """
# function contract(tr::MPSTensor{2},obj::MPSTensor{3})
#     return MPSTensor(@tensor tmp[-1,-2;-3] ≔ tr.A[-1,1] * obj.A[1,-2,-3])
# end

# function contract(obj::MPSTensor{3},tl::MPSTensor{2})
#     return MPSTensor(@tensor tmp[-1,-2;-3] ≔ obj.A[-1,-2,1] * tl.A[1,-3])
# end

# function contract(tl::MPSTensor{2},tr::MPSTensor{2})
#     return MPSTensor(@tensor tmp[-1,;-2] ≔ tl.A[-1,1] * tr.A[1,-2])
# end

# function contract(A::AdjointMPSTensor{2},B::MPSTensor{2})
#     return @tensor A.A[1,2] * B.A[2,1]
# end

# function contract(A::AdjointMPSTensor{3},B::MPSTensor{3})
#     return @tensor A.A[1,2,3] * B.A[2,3,1]
# end

# function contract(A::MPSTensor{3}, B::MPSTensor{3})
#     return _inproduct(A,adjoint(B))
# end

# function contract(A::MPSTensor{3}, B::AdjointMPSTensor{3})
#     return @tensor A.A[1,3,2] * B.A[2,1,3]
# end

# function contract(B::AdjointCompositeMPSTensor{2, 4}, A::CompositeMPSTensor{2, 4})
#     return @tensor A.A[1,3,4,2] * B.A[2,1,3,4]
# end

"""
scalar 
"""
# function contract(EnvL::SparseLeftEnvironmentTensor, A::MPSTensor{3}, B::SparseMPOTensor{N,M}, C::AdjointMPSTensor{3}, EnvR::SparseRightEnvironmentTensor) where {N,M}
#     tmp = 0
#     for i in 1:N, j in 1:M
#         isnothing(B.m[i,j]) && continue
#         @show typeof(B.m[i,j])
#         tmp += contract(EnvL.A[i], A, B.m[i,j], C, EnvR.A[j])
#     end
#     return tmp
# end
#= COMPOSITE ENVIRONMENT =#
# """
# ENVL + MPS
# make composite ENVL (ENVL + MPS 1) 
# """

# """
# ENVL + composite MPS 
# make composite ENVL (ENVL + composite MPS) 
# """
# function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, ::IdentityOperator{1})
#     @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,1] * A.A[1,-2,-3,-4]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, mpo::LocalOperator{1,1})
#     @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,1] * A.A[1,2,-3,-4] * mpo.A[-2,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, mpo::LocalOperator{1,2})
#     @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,1] * A.A[1,2,-3,-5] * mpo.A[-2,-4,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{3},A::CompositeMPSTensor{2, 4}, mpo::LocalOperator{2,1})
#     @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,3,1] * A.A[1,2,-3,-4] * mpo.A[-2,3,2]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftEnvironmentTensor{3},A::CompositeMPSTensor{2, 4}, ::IdentityOperator{1})
#     @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,-4,1] * A.A[1,-2,-3,-5]
#     return LeftCompositeEnvironmentTensor(tmp)
# end

# """
# composite ENV (ENVL + MPS 1) +
# make composite ENVL (ENVL + MPS 2)
# """
# function contract(El::LeftCompositeEnvironmentTensor{3,4}, mpo::LocalOperator{1,2})
#     @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,-2,1,-5] * mpo.A[-3,-4,1]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftCompositeEnvironmentTensor{3,5}, mpo::LocalOperator{2,1})
#     @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,-2,1,2,-4] * mpo.A[-3,2,1]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftCompositeEnvironmentTensor{3,4}, mpo::LocalOperator{1,1})
#     @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,-2,1,-4] * mpo.A[-3,1]
#     return LeftCompositeEnvironmentTensor(tmp)
# end
# function contract(El::LeftCompositeEnvironmentTensor{3,4}, ::IdentityOperator{1})
#     return El
# end
# function contract(El::LeftCompositeEnvironmentTensor{3, 5}, ::IdentityOperator{1})
#     return El
# end
# """
# composite ENV (ENVL + MPS 2) + ENVR
# make composite MPS
# """
# function contract(El::LeftCompositeEnvironmentTensor{3,4}, Er::RightEnvironmentTensor{2})
#     return CompositeMPSTensor(El.A*Er.A)
# end
# function contract(El::LeftCompositeEnvironmentTensor{3,5}, Er::RightEnvironmentTensor{3})
#     return CompositeMPSTensor(El.A*permute(Er.A,(2,1),(3,)))
# end

# """
# MPS + ENVR
# make composite ENVR (MPS + ENVR 1)
# """


#= EFFECTIVE MPS =#

"""
composite ENVL (ENVL + MPS 1) + composite ENVR (MPS + ENVR 1)
make eff composite MPS
"""
# function contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, EnvR::RightCompositeEnvironmentTensor{1, 3})
#     @tensor tmp[-1 -2 -3;-4] ≔ EnvL.A[-1,-2,1] * EnvR.A[1,-3,-4]
#     return CompositeMPSTensor(tmp)
# end

# function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightCompositeEnvironmentTensor{1, 4})
#     @tensor tmp[-1 -2 -3;-4] ≔ EnvL.A[-1,-2,2,1] * EnvR.A[1,2,-3,-4]
#     return CompositeMPSTensor(tmp)
# end

"""
composite ENVL (ENVL + MPS 1)+ ENVR
make eff MPS
"""

"""
r3 MPS + r2 MPS
connect a rank 3 MPS with a rank 2 MPS
"""

#= ================================================ =#

# function contract(EnvL::LeftEnvironmentTensor{2},A::Union{DenseMPOTensor{2},MPSTensor{2}})
#     return LeftEnvironmentTensor(@tensor tmp[-1;-2] ≔ EnvL.A[-1,1] * A.A[1,-2])
# end

# function contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightEnvironmentTensor{2})
#     return @tensor tmp[-1;-2] ≔ EnvL.A[-1,1] * EnvR.A[1,-2]
# end


"""
sparse ENVL + MPOs + sparse ENVR
make scalar
"""

"""
ENVL + MPO + ENVR
make scalar
"""
# function contract(EnvL::LeftEnvironmentTensor{2},A::MPSTensor{3},B::DenseMPOTensor{2},C::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[3,1] * A.A[1,2,5] * B.A[4,2] * C.A[6,3,4] * EnvR.A[5,6]
# end
# function contract(EnvL::LeftEnvironmentTensor{2},A::MPSTensor{3},B::LocalOperator{1,1},C::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[3,1] * A.A[1,2,5] * B.A[4,2] * C.A[6,3,4] * EnvR.A[5,6]
# end

# function contract(EnvL::LeftEnvironmentTensor{2},A::MPSTensor{3},::IdentityOperator{1},C::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[3,1] * A.A[1,2,4] * C.A[5,3,2] * EnvR.A[4,5]
# end

# function contract(EnvL::LeftEnvironmentTensor{2},A::MPSTensor{3},B::LocalOperator{1,2},C::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
#     return @tensor EnvL.A[3,1] * A.A[1,2,5] * B.A[4,6,2] * C.A[7,3,4] * EnvR.A[5,6,7]
# end

# function contract(EnvL::LeftEnvironmentTensor{3},A::MPSTensor{3},B::LocalOperator{2,1},C::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[4,3,1] * A.A[1,2,6] * B.A[5,2] * C.A[7,4,5] * EnvR.A[6,7]
# end

# function contract(EnvL::LeftEnvironmentTensor{3},A::Union{DenseMPOTensor{2},MPSTensor{2}})
#     return LeftEnvironmentTensor(@tensor tmp[-1;-2 -3] ≔ EnvL.A[-1,-2,1] * A.A[1,-3])
# end

# function contract(EnvL::LeftEnvironmentTensor{3},EnvR::RightEnvironmentTensor{3})
#     return @tensor tmp[-1;-2] ≔ EnvL.A[-1,2,1] * EnvR.A[1,2,-2]
# end

# function contract(EnvL::SparseLeftEnvironmentTensor, A::MPSTensor{3}, B::MPSTensor{3}, C::SparseMPOTensor{N₁,M₁}, D::SparseMPOTensor{N₂,M₂}, EnvR::SparseRightEnvironmentTensor) where {N₁,M₁,N₂,M₂}
#     @assert M₁ == N₂
#     tmp = nothing
#     for i in 1:N₁, j in 1:M₁, k in 1:M₂
#         isnothing(C.m[i,j]) | isnothing(D.m[j,k]) && continue
#         tmp1 = contract(EnvL.A[i], A, C.m[i,j])
#         tmp2 = contract(B, D.m[j,k], EnvR.A[k])
#         tmp = axpy!(1,contract(tmp1, tmp2),tmp)
#     end
#     return tmp
# end

# function contract(A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{2})
#     @tensor tmp[-1;-2] ≔ A.A[-1,3,1] * A′.A[2,-2,3] * EnvR.A[1,2]
#     return RightEnvironmentTensor(tmp)
# end

# function contract(EnvL::DenseLeftEnvironmentTensor, A::MPSTensor, B::AdjointMPSTensor, EnvR::DenseRightEnvironmentTensor)
#     return contract(EnvL.A, A, B, EnvR.A)
# end

# function contract(EnvL::LeftEnvironmentTensor{2}, A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{2})
#     return @tensor EnvL.A[1,2] * A.A[2,3,4] * A′.A[5,1,3] * EnvR.A[4,5]
# end




# function contract(A::MPSTensor{3}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{3})
#     @tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,2,1] * B.A[-3,2] * Er.A[1,-2,-4]
#     return RightCompositeEnvironmentTensor(tmp)
# end
