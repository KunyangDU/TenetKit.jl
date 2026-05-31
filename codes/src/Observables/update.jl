function dispatch!(node::DirectedNode, ::L2R)
    EnvL = node.val.EnvL
    leftdata = node.val.leftdata
    tasks = []
    for (e,c) in child(node)
        wt = e.weight
        wt_EnvL = deepcopy(EnvL)
        wt_leftdata = deepcopy(leftdata)
        edge_done = false
        lock(wt.lock) do
            if !isrightdefault(e)
                # (!isnothing(wt.EnvL) || !isnothing(wt.leftdata)) && return
                wt.EnvL = wt_EnvL
                wt.leftdata = wt_leftdata
                push!(tasks, wt)
                edge_done = true
                return
            elseif length(parent(c)) > 1
                # (!isnothing(wt.EnvL) || !isnothing(wt.leftdata)) && return
                wt.EnvL = wt_EnvL
                wt.leftdata = wt_leftdata
                edge_done = true
                return
            end
        end
        edge_done && continue
        c_val_EnvL = deepcopy(EnvL)
        c_val_leftdata = deepcopy(leftdata)
        ispush = false
        lock(c.val.lock) do
            # isnothing(c.val.EnvL) || return
            c.val.EnvL = c_val_EnvL
            c.val.leftdata = c_val_leftdata
            ispush = true
        end
        ispush && push!(tasks, c)
    end
    lock(node.val.lock) do
        leftdelfault_val!(node)
    end
    return tasks
end

function dispatch!(node::DirectedNode, ::R2L)
    EnvR = node.val.EnvR
    rightdata = node.val.rightdata
    tasks = []
    for (e,p) in parent(node)
        wt = e.weight
        wt_EnvR = deepcopy(EnvR)
        wt_rightdata = deepcopy(rightdata)
        edge_done = false
        lock(wt.lock) do
            if !isleftdefault(e)
                # (!isnothing(wt.EnvR) || !isnothing(wt.rightdata)) && return
                wt.EnvR = wt_EnvR
                wt.rightdata = wt_rightdata
                push!(tasks, wt)
                edge_done = true
                return
            elseif length(child(p)) > 1
                # (!isnothing(wt.EnvR) || !isnothing(wt.rightdata)) && return
                wt.EnvR = wt_EnvR
                wt.rightdata = wt_rightdata
                edge_done = true
                return
            end
        end
        edge_done && continue
        p_val_EnvR = deepcopy(EnvR)
        p_val_rightdata = deepcopy(rightdata)
        ispush = false
        lock(p.val.lock) do
            # isnothing(p.val.EnvR) || return
            p.val.EnvR = p_val_EnvR
            p.val.rightdata = p_val_rightdata
            ispush = true
        end
        ispush && push!(tasks, p)
    end
    lock(node.val.lock) do
        rightdelfault_val!(node)
    end
    return tasks
end

function _update!(node::DirectedNode, obj::T) where T <: Union{DenseMPS,DenseMPO}
    hasL, hasR = !isnothing(node.val.EnvL), !isnothing(node.val.EnvR)
    !hasL && !hasR && return Union{DirectedNode,ObservableWeight}[]
     hasL &&  hasR && return _update!(node, obj, C2C())
    !hasL &&  hasR && return _update!(node, obj, R2L())
     hasL && !hasR && return _update!(node, obj, L2R())
    throw(ErrorException("multi threading error"))
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
    ans = contract(weight.EnvL, weight.EnvR)
    names, sites = _lr_merge(weight.leftdata, weight.rightdata)
    default_weight!(weight)
    return (names, sites, ans)
end

function _update!(node::DirectedNode, obj::T, ::C2C) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    @assert 1 ≤ (site = _site(node.val)) ≤ L
    isnothing(node.val.EnvL) || isnothing(node.val.EnvR) && return nothing
    !reduce(&,node.val.isstring) && leftmergedata!(node.val)
    ans = contract(_calObs_left_contract(node.val, obj[site]), node.val.EnvR)
    names, sites = _lr_merge(node.val.leftdata, node.val.rightdata)
    default_val!(node)
    return (names, sites, ans)
end