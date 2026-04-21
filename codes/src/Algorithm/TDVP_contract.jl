"""
tanTRG tools
"""

function contract(B::AdjointMPOTensor{<:Number, 4}, A::DenseMPOTensor{<:Number, 4})
    return @tensor A.A[3,1,2,4] * B.A[2,4,3,1]
end

function contract(B::AdjointCompositeMPOTensor{<:Number, 2,6}, A::CompositeMPOTensor{<:Number, 2,6})
    return  @tensor A.A[5,6,2,1,3,4] * B.A[1,3,4,5,6,2]
end