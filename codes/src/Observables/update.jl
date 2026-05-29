function dispatch!(node::DirectedNode, ::L2R)
    # @assert isnothing(node.val.EnvR)
    # @assert !isnothing(node.val.EnvL)
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
                (!isnothing(wt.EnvL) || !isnothing(wt.leftdata)) && return
                wt.EnvL = wt_EnvL
                wt.leftdata = wt_leftdata
                push!(tasks, wt)
                edge_done = true
                return
            elseif length(parent(c)) > 1
                (!isnothing(wt.EnvL) || !isnothing(wt.leftdata)) && return
                wt.EnvL = wt_EnvL
                wt.leftdata = wt_leftdata
                # !isnothing(wt.EnvR) && push!(tasks, wt)
                edge_done = true
                return
            end
        end
        edge_done && continue
        c_val_EnvL = deepcopy(EnvL)
        c_val_leftdata = deepcopy(leftdata)
        should_push = false
        lock(c.val.lock) do
            isnothing(c.val.EnvL) || return
            c.val.EnvL = c_val_EnvL
            c.val.leftdata = c_val_leftdata
            should_push = true
        end
        should_push && push!(tasks, c)
    end
    lock(node.val.lock) do
        leftdelfault_val!(node)
    end
    return tasks
end

function dispatch!(node::DirectedNode, ::R2L)
    # @assert isnothing(node.val.EnvL)
    # @assert !isnothing(node.val.EnvR)
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
                (!isnothing(wt.EnvR) || !isnothing(wt.rightdata)) && return
                wt.EnvR = wt_EnvR
                wt.rightdata = wt_rightdata
                push!(tasks, wt)
                edge_done = true
                return
            elseif length(child(p)) > 1
                (!isnothing(wt.EnvR) || !isnothing(wt.rightdata)) && return
                wt.EnvR = wt_EnvR
                wt.rightdata = wt_rightdata
                # !isnothing(wt.EnvL) && push!(tasks, wt)
                edge_done = true
                return
            end
        end
        edge_done && continue
        p_val_EnvR = deepcopy(EnvR)
        p_val_rightdata = deepcopy(rightdata)
        should_push = false
        lock(p.val.lock) do
            isnothing(p.val.EnvR) || return
            p.val.EnvR = p_val_EnvR
            p.val.rightdata = p_val_rightdata
            should_push = true
        end
        should_push && push!(tasks, p)
    end
    lock(node.val.lock) do
        rightdelfault_val!(node)
    end
    return tasks
end

function _update!(node::DirectedNode, obj::T) where T <: Union{DenseMPS,DenseMPO}
    hasL, hasR = lock(node.val.lock) do
        !isnothing(node.val.EnvL), !isnothing(node.val.EnvR)
    end
    !hasL && !hasR && return Union{DirectedNode,ObservableWeight}[]
    hasL && hasR && return _update!(node, obj, C2C())
    !hasL && return _update!(node, obj, R2L())
    return _update!(node, obj, L2R())
end

function _update!(node::DirectedNode, obj::T, ::L2R) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    site, E = lock(node.val.lock) do
        site = node.val.A.site
        E = site ≥ 1 ? node.val.EnvL : nothing
        if !node.val.isstring
            push!(node.val.leftdata["name"], node.val.A.name)
            push!(node.val.leftdata["site"], node.val.A.site)
        end
        site, E
    end
    site ≥ 1 && (node.val.EnvL = contract(obj[site], node.val.A, obj[site]', E))
    return dispatch!(node, L2R())
end

function _update!(node::DirectedNode, obj::T, ::R2L) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    site, E = lock(node.val.lock) do
        site = node.val.A.site
        E = site ≤ L ? node.val.EnvR : nothing
        if !node.val.isstring
            push!(node.val.rightdata["name"], node.val.A.name)
            push!(node.val.rightdata["site"], node.val.A.site)
        end
        site, E
    end
    site ≤ L && (node.val.EnvR = contract(obj[site], node.val.A, obj[site]', E))
    return dispatch!(node, R2L())
end


function _update!(weight::ObservableWeight, ::T) where T <: Union{DenseMPS,DenseMPO}
    result = lock(weight.lock) do
        # isnothing(weight.EnvL) && isnothing(weight.EnvR) && default_weight!(weight)
        # (isnothing(weight.EnvL) && isnothing(weight.leftdata)) || (isnothing(weight.EnvR) && isnothing(weight.rightdata)) && return nothing
        (isnothing(weight.EnvL) || isnothing(weight.EnvR)) && return nothing
        wL, wR = weight.EnvL, weight.EnvR
        wld, wrd = deepcopy(weight.leftdata), deepcopy(weight.rightdata)
        default_weight!(weight)
        (wL, wR, wld, wrd)
    end
    result === nothing && return Union{DirectedNode,ObservableWeight}[]
    wL, wR, wld, wrd = result
    ans = contract(wL, wR)
    names, sites = _lr_merge(wld, wrd)
    return (names, sites, ans)
end

function _update!(node::DirectedNode, obj::T, ::C2C) where T <: Union{DenseMPS,DenseMPO}
    result = lock(node.val.lock) do
        isnothing(node.val.EnvL) || isnothing(node.val.EnvR) && return nothing
        site, isstr = node.val.A.site, node.val.isstring
        ld = isstr ? nothing : deepcopy(node.val.leftdata)
        rd = isstr ? nothing : deepcopy(node.val.rightdata)
        EL, ER = node.val.EnvL, node.val.EnvR
        default_val!(node)
        (site, EL, ER, isstr, ld, rd)
    end
    result === nothing && return Union{DirectedNode,ObservableWeight}[]
    site, EL, ER, isstr, ld, rd = result
    ans = contract(contract(obj[site], node.val.A, obj[site]', EL), ER)
    if !isstr
        names, sites = _lr_merge(ld, rd)
        return (names, sites, ans)
    end
    return (nothing, nothing, ans)  # isstring: no name tracking needed
end