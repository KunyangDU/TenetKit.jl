"""
DMRG tools
"""
function contract(tr::MPSTensor{<:Number, 2},obj::MPSTensor{<:Number, 3})
    return MPSTensor(@tensor tmp[-1,-2;-3] ≔ tr.A[-1,1] * obj.A[1,-2,-3])
end

function contract(obj::MPSTensor{<:Number, 3},tl::MPSTensor{<:Number, 2})
    return MPSTensor(@tensor tmp[-1,-2;-3] ≔ obj.A[-1,-2,1] * tl.A[1,-3])
end

function contract(tl::MPSTensor{<:Number, 2},tr::MPSTensor{<:Number, 2})
    return MPSTensor(@tensor tmp[-1,;-2] ≔ tl.A[-1,1] * tr.A[1,-2])
end

function contract(A::AdjointMPSTensor{<:Number, 2},B::MPSTensor{<:Number, 2})
    return @tensor A.A[1,2] * B.A[2,1]
end

function contract(A::AdjointMPSTensor{<:Number, 3},B::MPSTensor{<:Number, 3})
    return @tensor A.A[1,2,3] * B.A[2,3,1]
end

function contract(A::MPSTensor{<:Number, 3}, B::MPSTensor{<:Number, 3})
    return _inproduct(A,adjoint(B))
end

function contract(A::MPSTensor{<:Number, 3}, B::AdjointMPSTensor{<:Number, 3})
    return @tensor A.A[1,3,2] * B.A[2,1,3]
end

function contract(B::AdjointCompositeMPSTensor{<:Number, 2, 4}, A::CompositeMPSTensor{<:Number, 2, 4})
    return @tensor A.A[1,3,4,2] * B.A[2,1,3,4]
end

