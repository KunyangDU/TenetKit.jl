function dispatch!(node::DirectedNode, ::L2R)
    EnvL,leftdata = lock(node.val.lock) do
        node.val.EnvL, node.val.leftdata
    end
    tasks = []
    for (e,c) in child(node)
        leftdata′ = deepcopy(leftdata)
        wt = e.weight
        edge_done = false
        lock(wt.lock) do
            if !isrightdefault(e)
                wt.EnvL = EnvL
                wt.leftdata = leftdata′
                push!(tasks, wt)
                edge_done = true
                return
            elseif length(parent(c)) > 1
                wt.EnvL = EnvL
                wt.leftdata = leftdata′
                edge_done = true
                return
            end
        end
        edge_done && continue
        lock(c.val.lock) do
            c.val.EnvL = EnvL * wt.strength
            c.val.leftdata = leftdata′
        end
        push!(tasks, c)
    end
    lock(node.val.lock) do
        isrightdefault(node.val) && leftdelfault_val!(node)
    end
    return tasks
end

function dispatch!(node::DirectedNode, ::R2L)
    EnvR,rightdata = lock(node.val.lock) do
        node.val.EnvR, node.val.rightdata
    end
    tasks = []
    for (e,p) in parent(node)
        rightdata′ = deepcopy(rightdata)
        wt = e.weight
        edge_done = false
        lock(wt.lock) do
            if !isleftdefault(e)
                wt.EnvR = EnvR
                wt.rightdata = rightdata′
                push!(tasks, wt)
                edge_done = true
                return
            elseif length(child(p)) > 1
                wt.EnvR = EnvR
                wt.rightdata = rightdata′
                edge_done = true
                return
            end
        end
        edge_done && continue
        lock(p.val.lock) do
            p.val.EnvR = EnvR * wt.strength
            p.val.rightdata = rightdata′
        end
        push!(tasks, p)
    end
    lock(node.val.lock) do
        isleftdefault(node.val) && rightdelfault_val!(node)
    end
    return tasks
end

function _update!(node::DirectedNode, obj::T) where T <: Union{DenseMPS,DenseMPO}
    hasL, hasR = !isnothing(node.val.EnvL), !isnothing(node.val.EnvR)
    !hasL && !hasR && return Union{DirectedNode,ObservableWeight}[]
     hasL &&  hasR && return _update!(node, obj, C2C())
    !hasL &&  hasR && return _update!(node, obj, R2L())
     hasL && !hasR && return _update!(node, obj, L2R())
end

function _update!(node::DirectedNode, obj::T, ::L2R) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    !reduce(&,node.val.isstring) && leftmergedata!(node.val)
    (site = _site(node.val)) ≥ 1 && (node.val.EnvL = _calObs_left_contract(node.val,obj[site]))
    return L2R()
end

function _update!(node::DirectedNode, obj::T, ::R2L) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    !reduce(&,node.val.isstring) && rightmergedata!(node.val)
    (site = _site(node.val)) ≤ L && (node.val.EnvR = _calObs_right_contract(node.val,obj[site]))
    return R2L()
end


function _update!(weight::ObservableWeight, ::T) where T <: Union{DenseMPS,DenseMPO}
    (isnothing(weight.EnvL) || isnothing(weight.EnvR)) && return nothing
    ans = contract(weight.EnvL, weight.EnvR) * weight.strength
    names, sites = _lr_merge(weight.leftdata, weight.rightdata)
    default_weight!(weight)
    return (names, sites, ans)
end

function _update!(node::DirectedNode, obj::T, ::C2C) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    @assert 1 ≤ (site = _site(node.val)) ≤ L
    isnothing(node.val.EnvL) || isnothing(node.val.EnvR) && return nothing
    
    @assert (lsite = _leftsite(node.val)) ≠ (rsite = _rightsite(node.val)) "L2R + R2L on same node"
    if site == lsite || site == rsite
        ans = contract(node.val.EnvL, node.val.EnvR)
    else
        !reduce(&,node.val.isstring) && leftmergedata!(node.val)
        ans = contract(_calObs_left_contract(node.val, obj[site]), node.val.EnvR)
    end
    names, sites = _lr_merge(node.val.leftdata, node.val.rightdata)
    default_val!(node)
    return (names, sites, ans)
end