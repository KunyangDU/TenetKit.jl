function default_val!(node::DirectedNode{ObservableOperator{R1,R2}}) where {R1,R2}
    leftdelfault_val!(node)
    rightdelfault_val!(node)
end

function leftdelfault_val!(node::DirectedNode{ObservableOperator{R1,R2}}) where {R1,R2}
    node.val.EnvL = nothing
    node.val.leftdata = nothing
end

function rightdelfault_val!(node::DirectedNode{ObservableOperator{R1,R2}}) where {R1,R2}
    node.val.EnvR = nothing
    node.val.rightdata = nothing
end

isdefault(node::DirectedNode{ObservableOperator{R1,R2}}) where {R1,R2} = isnothing(node.val.EnvL) && isnothing(node.val.EnvR) && isnothing(node.val.leftdata) && isnothing(node.val.rightdata)

_lr_merge(left::Dict,right::Dict) = map(x -> tuple(left[x]..., reverse(right[x])...), ["name","site"])

contract(EnvL::LeftEnvironmentTensor{2},EnvR::RightEnvironmentTensor{2}) = @tensor EnvL.A[1,2] * EnvR.A[2,1]
contract(EnvL::LeftEnvironmentTensor{3},EnvR::RightEnvironmentTensor{3}) = @tensor EnvL.A[1,2,3] * EnvR.A[3,2,1]

dictsize(d::Dict) = sum(v -> v isa Dict ? dictsize(v) : 1, values(d))

function deepmerge!(d1::Dict, d2::Dict)
    for (k, v2) in d2
        if haskey(d1, k) && d1[k] isa Dict && v2 isa Dict
            deepmerge!(d1[k], v2)
        else
            d1[k] = v2
        end
    end
    return d1
end
