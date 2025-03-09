function _cbeorth!(Q::DenseMPOTensor{4},A::DenseMPOTensor{4},direction::Symbol)
    Q.A -= _cbeproj(Q,A,direction)
end

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

function _cbeinner(Q::MPSTensor{3},A::MPSTensor{3},direction::Symbol)
    if direction == :right
        return @tensor tmp[-1;-2] ≔ Q.A[-1,2,1] * A'.A[1,-2,2]
    elseif direction == :left
        return @tensor tmp[-1;-2] ≔ Q.A[1,2,-2] * A'.A[-1,1,2]
    end
end

