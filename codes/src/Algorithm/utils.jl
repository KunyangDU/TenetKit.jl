
function _cbeproj(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},direction::Symbol)
    if direction == :right
        return @tensor tmp[-1,-2;-3,-4] ≔ Q.A[2,-2,1,3] * A'.A[1,3,2,4] * A.A[-1,4,-3,-4]
    elseif direction == :left 
        return @tensor tmp[-1,-2;-3,-4] ≔ Q.A[2,1,-3,3] * A'.A[4,3,2,1] * A.A[-1,-2,4,-4]
    end
end

function _cbeinner(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},direction::Symbol)
    if direction == :right
        return @tensor tmp[-1;-2] ≔ Q.A[2,-1,1,3] * A'.A[1,3,2,-2]
    elseif direction == :left
        return @tensor tmp[-1;-2] ≔ Q.A[2,1,-2,3] * A'.A[-1,3,2,1]
    end
end

function _cbeproj(Q::MPSTensor{3},A::MPSTensor{3},direction::Symbol)
    if direction == :right
        return @tensor tmp[-1,-2;-3] ≔ Q.A[-1,2,1] * A'.A[1,3,2] * A.A[3,-2,-3]
    elseif direction == :left 
        return @tensor tmp[-1,-2;-3] ≔ Q.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3]
    end
end

function _cbeinner(Q::MPSTensor{3},A::MPSTensor{3},direction::Symbol)
    if direction == :right
        return @tensor tmp[-1;-2] ≔ Q.A[-1,2,1] * A'.A[1,-2,2]
    elseif direction == :left
        return @tensor tmp[-1;-2] ≔ Q.A[1,2,-2] * A'.A[-1,1,2]
    end
end

_expanddim(::ComplexSpace,D::Int64) = ℂ^D

function _expanddim(S::GradedSpace,D::Int64)
    ratio = D / dim(S)
    for (c,d) in S.dims
        S.dims[c] = ceil(Int64,d*ratio)
    end
    return S
end

function _cbetensor(func,A::MPSTensor{3}, D_f::Int64, direction::Symbol)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if direction == :left
        tmp = MPSTensor(func,cdm,_expanddim(fuse(cdm),D_f))
    elseif direction == :right
        tmp = MPSTensor(func,_expanddim(fuse((cdm[2] ⊗ dm)),D_f) ⊗ cdm[2],dm)
    end
    normalize!(tmp)
    return tmp'
end

function _cbetensor(func,A::DenseMPOTensor{4}, D_f::Int64,direction::Symbol)
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    if direction == :left
        tmp = DenseMPOTensor(func,cdm,(_expanddim(fuse(cdm⊗dm[2]),D_f) )⊗dm[2])
    elseif direction == :right
        tmp = DenseMPOTensor(func,cdm[1]⊗(_expanddim(fuse(dm⊗cdm[1]),D_f)),dm)
    end
    normalize!(tmp)
    return tmp'
end

function _cbedsum(Q::MPSTensor{3},A::MPSTensor{3},direction::Symbol)
    if direction == :right
        ~,Q = rightorth(catcodomain(map(x -> permute(x.A,(1,),(2,3)),(Q,A))...))
        Q = MPSTensor(permute(Q,(1,2),(3,)))
    elseif direction == :left 
        Q,~ = leftorth(catdomain(Q.A,A.A))
        Q = MPSTensor(Q)
    end
    return Q
end

function _cbedsum(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},direction::Symbol)
    if direction == :right
        Q = catcodomain(map(x -> permute(x.A,(2,),(1,3,4)),(Q,A))...)
        ~,Q = rightorth(DenseMPOTensor(permute(Q,(2,1),(3,4))))
    elseif direction == :left 
        Q = catdomain(map(x -> permute(x.A,(1,2,4),(3,)),(Q,A))...)
        Q,~ = leftorth(DenseMPOTensor(permute(Q,(1,2),(4,3))))
    end
    return Q
end


