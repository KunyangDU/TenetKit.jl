hasLR(node::DirectedNode) = !isleftdefault(node.val), !isrightdefault(node.val)
hasLR(nodeL::DirectedNode, nodeR::DirectedNode) = hasLR(nodeL)..., hasLR(nodeR)...
function dispatch!(nodeL::DirectedNode, nodeR::DirectedNode)
    (isdefault(nodeL) || isdefault(nodeR)) && return DirectedEdge[]
    tasks = DirectedEdge[]
    LhasL,LhasR,RhasL,RhasR = hasLR(nodeL,nodeR)
    @assert LhasL ⊻ LhasR "node L overlap!",nodeL
    @assert RhasL ⊻ RhasR "node R overlap!",nodeR
    !LhasL && !RhasR && return DirectedEdge[]
    @assert (LhasL && RhasL) || (LhasR && RhasR) LhasL,LhasR,RhasL,RhasR

    if LhasL && RhasL
        # push!(tasks, filter(x -> length(parent(x.to)) == 1,nodeR.out_edges)...)
        push!(tasks, nodeR.out_edges...)
        reduce(&,map(x -> !isleftdefault(x[2].val), child(nodeL))) && leftdelfault_val!(nodeL)
    else
        # push!(tasks, filter(x -> length(child(x.from)) == 1,nodeL.in_edges)...)
        push!(tasks, nodeL.in_edges...)
        reduce(&,map(x -> !isrightdefault(x[2].val), parent(nodeR))) && rightdelfault_val!(nodeR)
    end
    return tasks
end

function _update!(edge::DirectedEdge{T₁},obj::T₂) where {T₁ <: Union{ObservableWeight, CompositeObservableOperator}, T₂ <: Union{DenseMPS,DenseMPO}}
    from, to = edge.from, edge.to
    tasks = nothing
    lock(from.val.lock) do 
        lock(to.val.lock) do 
            tasks = _update!(from, edge.weight, to, from.val.EnvL, to.val.EnvR, obj)
            isnothing(tasks) && (tasks = dispatch!(from, to))
        end
    end
    return tasks
end

function _update!(nodeL::DirectedNode, weight::ObservableWeight, nodeR::DirectedNode, ::LeftEnvironmentTensor, ::Nothing, obj::T₂) where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L
    length(parent(nodeR)) > 1 && return nothing
    nodeR.val.leftdata = deepcopy(nodeL.val.leftdata)
    nodeR.val.EnvL = nodeL.val.EnvL
    !reduce(&,nodeR.val.isstring) && leftmergedata!(nodeR.val)
    (site = _site(nodeR.val)) ≤ L && (nodeR.val.EnvL = _calObs_left_contract(nodeR.val,obj[site]) * weight.strength)
    return nothing
end

function _update!(nodeL::DirectedNode, weight::ObservableWeight, nodeR::DirectedNode, ::Nothing, ::RightEnvironmentTensor, obj::T₂) where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L
    length(child(nodeL)) > 1 && return nothing
    nodeL.val.rightdata = deepcopy(nodeR.val.rightdata)
    nodeL.val.EnvR = nodeR.val.EnvR
    !reduce(&,nodeL.val.isstring) && rightmergedata!(nodeL.val)
    (site = _site(nodeL.val)) ≥ 1 && (nodeL.val.EnvR = _calObs_right_contract(nodeL.val,obj[site]) * weight.strength)
    return nothing
end

function _update!(nodeL::DirectedNode, weight::ObservableWeight, nodeR::DirectedNode, ::LeftEnvironmentTensor, ::RightEnvironmentTensor, obj::T₂) where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L
    names, sites = _lr_merge(nodeL.val.leftdata, nodeR.val.rightdata)
    ans = contract(nodeL.val.EnvL, nodeR.val.EnvR) * weight.strength
    return (names,sites,ans)
end

_update!(nodeL::DirectedNode, weight::ObservableWeight, nodeR::DirectedNode, ::Nothing, ::Nothing, obj::T₂) where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L = nothing
