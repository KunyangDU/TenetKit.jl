
function dispatch!(edge::DirectedEdge)
    tasks = DirectedEdge[]
    nodeL,nodeR = edge.from,edge.to
    hasL,hasR = hasLR(edge)
    !hasR && length(parent(nodeR)) > 1 && return DirectedEdge[]
    !hasL && length(child(nodeL)) > 1 && return DirectedEdge[]
    !hasL && !hasR && return DirectedEdge[]
    @assert hasL ⊻ hasR hasL,hasR
    if hasL && !hasR
        for e in nodeR.out_edges
            e.weight.EnvL = edge.weight.EnvL
            e.weight.leftdata = deepcopy(edge.weight.leftdata)
            push!(tasks, e)
        end
        leftdelfault_weight!(edge)
    elseif !hasL && hasR
        for e in nodeL.in_edges
            e.weight.EnvR = edge.weight.EnvR
            e.weight.rightdata = deepcopy(edge.weight.rightdata)
            push!(tasks, e)
        end
        rightdelfault_weight!(edge)
    end
    return tasks
end

function _update!(edge::DirectedEdge{T₁},obj::T₂) where {T₁ <: Union{ObservableWeight, CompositeObservableOperator}, T₂ <: Union{DenseMPS,DenseMPO}}
    from, to = edge.from, edge.to
    tasks = nothing
    lock(from.val.lock) do 
        lock(to.val.lock) do 
            tasks = _update!(from, edge.weight, to, edge.weight.EnvL, edge.weight.EnvR, obj)
            isnothing(tasks) && (tasks = dispatch!(edge))
        end
    end
    return tasks
end

function _update!(::DirectedNode, weight::ObservableWeight, nodeR::DirectedNode, ::LeftEnvironmentTensor, ::Nothing, obj::T₂) where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L
    length(parent(nodeR)) > 1 && return nothing
    !reduce(&,nodeR.val.isstring) && leftmergedata!(weight,nodeR.val)
    (site = _site(nodeR.val)) ≤ L && _calObs_left_contract!(weight, nodeR.val, obj[site])
    return nothing
end

function _update!(nodeL::DirectedNode, weight::ObservableWeight, ::DirectedNode, ::Nothing, ::RightEnvironmentTensor, obj::T₂) where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L
    length(child(nodeL)) > 1 && return nothing
    !reduce(&,nodeL.val.isstring) && rightmergedata!(weight,nodeL.val)
    (site = _site(nodeL.val)) ≥ 1 && _calObs_right_contract!(weight, nodeL.val,obj[site])
    return nothing
end

function _update!(::DirectedNode, weight::ObservableWeight, ::DirectedNode, ::LeftEnvironmentTensor, ::RightEnvironmentTensor, obj::T₂) where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L
    names, sites = _lr_merge(weight.leftdata, weight.rightdata)
    ans = contract(weight.EnvL, weight.EnvR) * weight.strength
    default_weight!(weight,false)
    return (names,sites,ans)
end

_update!(::DirectedNode, ::ObservableWeight, ::DirectedNode, ::Nothing, ::Nothing, ::T₂) where T₂ <: Union{DenseMPS{L},DenseMPO{L}} where L = nothing
