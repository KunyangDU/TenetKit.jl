"""
DMRG tools
"""
function contract(tr::MPSTensor{2},obj::MPSTensor{3})
    return MPSTensor(@tensor tmp[-1,-2;-3] ≔ tr.A[-1,1] * obj.A[1,-2,-3])
end

function contract(obj::MPSTensor{3},tl::MPSTensor{2})
    return MPSTensor(@tensor tmp[-1,-2;-3] ≔ obj.A[-1,-2,1] * tl.A[1,-3])
end

function contract(tl::MPSTensor{2},tr::MPSTensor{2})
    return MPSTensor(@tensor tmp[-1,;-2] ≔ tl.A[-1,1] * tr.A[1,-2])
end

function contract(A::AdjointMPSTensor{2},B::MPSTensor{2})
    return @tensor A.A[1,2] * B.A[2,1]
end

function contract(A::AdjointMPSTensor{3},B::MPSTensor{3})
    return @tensor A.A[1,2,3] * B.A[2,3,1]
end

function contract(A::MPSTensor{3}, B::MPSTensor{3})
    return _inproduct(A,adjoint(B))
end

function contract(A::MPSTensor{3}, B::AdjointMPSTensor{3})
    return @tensor A.A[1,3,2] * B.A[2,1,3]
end

function contract(B::AdjointCompositeMPSTensor{2, 4}, A::CompositeMPSTensor{2, 4})
    return @tensor A.A[1,3,4,2] * B.A[2,1,3,4]
end

