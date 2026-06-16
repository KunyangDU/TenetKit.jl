function dispatch!(node::DirectedNode, ::L2R)
    tasks = []
    EnvL,leftdata,hasL,hasR = lock(node.val.lock) do
        node.val.EnvL, node.val.leftdata, !isleftdefault(node.val), !isrightdefault(node.val)
    end
    # @show node,hasR,node.val.EnvR,node.val.rightdata
    hasR && return tasks
    !hasL && return tasks
    for (e,c) in child(node)
        leftdata′ = deepcopy(leftdata)
        wt = e.weight
        edge_done = false
        lock(c.val.lock) do
            lock(wt.lock) do
                if !isrightdefault(e)
                    wt.EnvL = EnvL
                    wt.leftdata = leftdata′
                    push!(tasks, wt)
                    edge_done = true
                elseif length(parent(c)) > 1 || (length(child(node)) > 1 && !isrightdefault(c.val))
                    wt.EnvL = EnvL
                    wt.leftdata = leftdata′
                    edge_done = true
                end
            end
            if !edge_done
                # if length(child(node)) == 1 && !isleftdefault(node.val) && !isrightdefault(c.val)
                #     @show "conflict",node.val,c.val
                #     return
                # end 
                # c.val.EnvL = EnvL * wt.strength
                # c.val.leftdata = leftdata′
                # push!(tasks, c)
                if length(child(node)) == 1 && !isleftdefault(node.val) && !isrightdefault(c.val)
                    rightdelfault_val!(c)
                    @show "conflict! L2R",node.val,c.val
                    while isrightdefault(node.val)
                        sleep(1e-1)
                        @show "wait L2R",node.val,c.val
                    end
                end
                if isleftdefault(node.val) || isrightdefault(c.val)
                    if isleftdefault(node.val)
                        @show "collect",node.val,c.val
                    end
                    c.val.EnvL = EnvL * wt.strength
                    c.val.leftdata = leftdata′
                    push!(tasks, c)
                end
            end
        end
    end
    lock(node.val.lock) do
        # if (isleftdefault(node.val) || isrightdefault(node.val))
        #     default_val!(node)
        # end
        # isrightdefault(node.val) && leftdelfault_val!(node)
    end
    return tasks
end

function dispatch!(node::DirectedNode, ::R2L)
    tasks = []
    EnvR,rightdata,hasL,hasR = lock(node.val.lock) do
        node.val.EnvR, node.val.rightdata, !isleftdefault(node.val), !isrightdefault(node.val)
    end
    # @show node,hasL,node.val.EnvL,node.val.leftdata
    hasL && return tasks
    !hasR && return tasks
    for (e,p) in parent(node)
        rightdata′ = deepcopy(rightdata)
        wt = e.weight
        edge_done = false
        lock(p.val.lock) do
            lock(wt.lock) do
                if !isleftdefault(e)
                    wt.EnvR = EnvR
                    wt.rightdata = rightdata′
                    push!(tasks, wt)
                    edge_done = true
                elseif length(child(p)) > 1 || (length(parent(node)) > 1 && !isleftdefault(p.val))
                    wt.EnvR = EnvR
                    wt.rightdata = rightdata′
                    edge_done = true
                end
            end
            if !edge_done
                # if length(parent(node)) == 1 && !isleftdefault(p.val) && !isrightdefault(node.val)
                #     @show "conflict",node.val,p.val
                #     return
                # end 
                # p.val.EnvR = EnvR * wt.strength
                # p.val.rightdata = rightdata′
                # push!(tasks, p)
                if length(parent(node)) == 1 && !isleftdefault(p.val) && !isrightdefault(node.val)
                    leftdelfault_val!(p)
                    @show "conflict! R2L",node.val,p.val
                    while isleftdefault(node.val)
                        sleep(1e-1)
                        @show "wait R2L",node.val,p.val
                    end
                end
                if isleftdefault(p.val) || isrightdefault(node.val)
                    if isrightdefault(node.val)
                        @show "collect",node.val,p.val
                    end
                    p.val.EnvR = EnvR * wt.strength
                    p.val.rightdata = rightdata′
                    push!(tasks, p)
                end
            end
        end
    end
    lock(node.val.lock) do
        # if (isleftdefault(node.val) || isrightdefault(node.val))
        #     default_val!(node)
        # end
        # isleftdefault(node.val) && rightdelfault_val!(node)
    end
    return tasks
end

function _update!(node::DirectedNode, obj::T) where T <: Union{DenseMPS,DenseMPO}
    hasL, hasR = !isnothing(node.val.EnvL), !isnothing(node.val.EnvR)
    !hasL && !hasR && return (@show "nothing!",node;Union{DirectedNode,ObservableWeight}[])
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
    if isnothing(weight.EnvL) || isnothing(weight.EnvR)
        @show weight.EnvL, weight.EnvR
        default_weight!(weight,false)
        return nothing
    end
    # @show _leftsite(weight), _rightsite(weight)
    ans = contract(weight.EnvL, weight.EnvR) * weight.strength
    names, sites = _lr_merge(weight.leftdata, weight.rightdata)
    default_weight!(weight,false)
    return (names, sites, ans)
end

function _update!(node::DirectedNode, obj::T, ::C2C) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    @assert 1 ≤ (site = _site(node.val)) ≤ L
    if isnothing(node.val.EnvL) || isnothing(node.val.EnvR)
        @show site,node.val.EnvL,node.val.EnvR
        default_val!(node)
        return nothing
    end
    
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