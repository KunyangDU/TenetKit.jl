function default_val!(node::DirectedNode{ObservableOperator{R1,R2}}) where {R1,R2}
    node.val.EnvL = nothing
    node.val.EnvR = nothing
    node.val.leftdata = nothing
    node.val.rightdata = nothing
end

_lr_merge(left::Dict,right::Dict) = map(x -> tuple(left[x]..., reverse(right[x])...), ["name","site"])

contract(EnvL::LeftEnvironmentTensor{2},EnvR::RightEnvironmentTensor{2}) = @tensor EnvL.A[1,2] * EnvR.A[2,1]
contract(EnvL::LeftEnvironmentTensor{3},EnvR::RightEnvironmentTensor{3}) = @tensor EnvL.A[1,2,3] * EnvR.A[3,2,1]

dictsize(d::Dict) = sum(v -> v isa Dict ? dictsize(v) : 1, values(d))
