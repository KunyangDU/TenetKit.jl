"""
axpby!
"""
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,1] * A.A[-1,1,2,-4] * EnvR.A[2,-3]
    return DenseMPOTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,1] * A.A[-2,1,2,-6] * B.A[-1,2,3,-5] * EnvR.A[3,-4]
    return CompositeMPOTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,-1,1,3] * B.A[2,3,4,-2] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[3,-1,1,5] * B.A[6,-2,2,3] * C.A[4,5,6,-3] * EnvR.A[1,2,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,2,-2,3] * B.A[-1,3,4,1] * EnvL.A[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1 ;-2 -3] ≔ A.A[6,4,-3,5] * B.A[3,2,-2,6] * C.A[-1,5,3,1] * EnvL.A[1,2,4]
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvL::DenseLeftEnvironmentTensor, A::DenseMPOTensor, B::DenseMPOTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end
function contract(EnvL::DenseLeftEnvironmentTensor, A::DenseMPOTensor, B::AdjointMPOTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end

"""
mul!
"""

function contract(El::LeftCompositeEnvironmentTensor{2, 4}, Er::RightCompositeEnvironmentTensor{2, 4})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ El.A[-3,-2,1,-6] * Er.A[1,-1,-4,-5]
    return CompositeMPOTensor(tmp)
end

function contract(El::LeftCompositeEnvironmentTensor{2, 5}, Er::RightCompositeEnvironmentTensor{2, 5})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ El.A[-3,-2,2,1,-6] * Er.A[1,2,-1,-4,-5]
    return CompositeMPOTensor(tmp)
end

function contract(EnvL::SparseLeftEnvironmentTensor, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::SparseMPOTensor{N₁,M₁}, D::SparseMPOTensor{N₂,M₂}, EnvR::SparseRightEnvironmentTensor) where {N₁,M₁,N₂,M₂}
    @assert M₁ == N₂
    tmp = nothing
    for i in 1:N₁, j in 1:M₁, k in 1:M₂
        isnothing(C.m[i,j]) | isnothing(D.m[j,k]) && continue
        tmp1 = contract(EnvL.A[i], A, C.m[i,j])
        tmp2 = contract(B, D.m[j,k], EnvR.A[k])
        if isnothing(tmp)
            tmp = contract(tmp1, tmp2)
        else
            tmp += contract(tmp1, tmp2)
        end
    end
    return tmp
end

function contract(EnvL::SparseLeftEnvironmentTensor, A::MPSTensor{3}, B::MPSTensor{3}, C::SparseMPOTensor{N₁,M₁}, D::SparseMPOTensor{N₂,M₂}, EnvR::SparseRightEnvironmentTensor) where {N₁,M₁,N₂,M₂}
    @assert M₁ == N₂
    tmp = nothing
    for i in 1:N₁, j in 1:M₁, k in 1:M₂
        isnothing(C.m[i,j]) | isnothing(D.m[j,k]) && continue
        tmp1 = contract(EnvL.A[i], A, C.m[i,j])
        tmp2 = contract(B, D.m[j,k], EnvR.A[k])
        tmp = axpy!(1,contract(tmp1, tmp2),tmp)
    end
    return tmp
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, EnvR::RightCompositeEnvironmentTensor{1, 3})
    @tensor tmp[-1 -2 -3;-4] ≔ EnvL.A[-1,-2,1] * EnvR.A[1,-3,-4]
    return CompositeMPSTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightCompositeEnvironmentTensor{1, 4})
    @tensor tmp[-1 -2 -3;-4] ≔ EnvL.A[-1,-2,2,1] * EnvR.A[1,2,-3,-4]
    return CompositeMPSTensor(tmp)
end


function contract(tr::DenseMPOTensor{2},obj::DenseMPOTensor{4})
    return DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ tr.A[-2,1] * obj.A[-1,1,-3,-4])
end

function contract(obj::DenseMPOTensor{4},tl::DenseMPOTensor{2})
    return DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ obj.A[-1,-2,1,-4] * tl.A[1,-3])
end

"""
make scalar
"""
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[2,1] * A.A[3,1,5,4] * B.A[6,4,3,2] * EnvR.A[5,6]
end

"""
densify
"""

function contract(EnvL::LeftCompositeEnvironmentTensor{3, 6}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,-2,-1,1,-5,-6] * EnvR.A[1,-4]
    return CompositeMPOTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{3, 7}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,-2,-1,2,1,-5,-6] * EnvR.A[1,2,-4]
    return CompositeMPOTensor(tmp)
end


function contract(EnvL::DenseLeftEnvironmentTensor{3}, tL::DenseMPOTensor{4}, tR::DenseMPOTensor{4}, hl::DenseMPOTensor{4}, hr::DenseMPOTensor{4}, EnvR::DenseRightEnvironmentTensor{3})
    @tensor tmp[-3,-2,-1;-4,-5,-6] ≔ EnvL.A.A[-1,1,2] * tL.A[3,2,7,-6] * tR.A[6,7,4,-5] * hl.A[-2,1,8,3] * hr.A[-3,8,5,6] * EnvR.A.A[4,5,-4]
    # @tensor tmp[-3,-2,-1;-4,-5,-6] ≔ EnvL.A.A[-1,1,2] * tL.A[3,2,5,-6] * tR.A[6,5,7,-5] * hl.A[-2,1,4,3] * hr.A[-3,4,8,6] * EnvR.A.A[7,8,-4]
    # @tensor tmp[-1,-2,-3,-6,-7;-5,-4] ≔ EnvL.A.A[-1,1,2] * tL.A[3,2,5,-7] * tR.A[6,5,-5,-6] * hl.A[-2,1,4,3] * hr.A[-3,4,-4,6]
    # return CompositeMPOTensor(permute(tmp * EnvR.A.A,(3,2,1),(6,4,5)))
    return CompositeMPOTensor(tmp)
end

function contract(EnvL::DenseLeftEnvironmentTensor{3}, tL::DenseMPOTensor{4}, tR::DenseMPOTensor{4}, hl::AdjointMPOTensor{4}, hr::AdjointMPOTensor{4}, EnvR::DenseRightEnvironmentTensor{3})
    # @tensor tmp[-1,-2,-3,-4,-5,-6,-7] ≔ EnvL.A.A[-1,1,2] * tL.A[3,2,5,-7] * tR.A[6,5,-5,-6] * hl.A[4,-2,3,1] * hr.A[-4,-3,6,4]
    # @tensor ft[-1,-2,-3;-4,-5,-6] ≔ tmp[-3,-2,-1,1,2,-5,-6] * EnvR.A.A[2,1,-4]
    # @tensor ft[-3,-2,-1;-4,-5,-6] ≔ EnvL.A.A[-1,1,2] * tL.A[3,2,5,-6] * tR.A[6,5,7,-5] * hl.A[4,-2,3,1] * hr.A[8,-3,6,4] * EnvR.A.A[7,8,-4]
    @tensor ft[-3,-2,-1;-4,-5,-6] ≔ EnvL.A.A[-1,1,2] * tL.A[3,2,7,-6] * tR.A[6,7,4,-5] * hl.A[8,-2,3,1] * hr.A[5,-3,6,8] * EnvR.A.A[4,5,-4]
    return CompositeMPOTensor(ft)
end