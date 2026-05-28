function dispatch!(node::DirectedNode, ::L2R)
    @assert isnothing(node.val.EnvR)
    @assert !isnothing(node.val.EnvL)
    EnvL = node.val.EnvL
    leftdata = node.val.leftdata
    tasks = []
    for (e,c) in child(node)
        if !isdefault(e)
            @assert isnothing(e.weight.EnvL) && isnothing(e.weight.leftdata)
            @assert !isnothing(e.weight.EnvR) && !isnothing(e.weight.rightdata)
            e.weight.EnvL = EnvL
            e.weight.leftdata = deepcopy(leftdata)
            push!(tasks, e.weight)
            continue
        elseif length(parent(c)) > 1
            e.weight.EnvL = EnvL
            e.weight.leftdata = deepcopy(leftdata)
            continue 
        end
        c.val.EnvL = deepcopy(EnvL)
        c.val.leftdata = deepcopy(leftdata)
        @assert !isnothing(c.val.EnvL)
        isnothing(c.val.EnvR) && push!(tasks, c)
    end
    node.val.EnvL = nothing
    node.val.leftdata = nothing
    return tasks
end

function dispatch!(node::DirectedNode, ::R2L)
    @assert isnothing(node.val.EnvL)
    @assert !isnothing(node.val.EnvR)
    EnvR = node.val.EnvR
    rightdata = node.val.rightdata
    tasks = []
    for (e,p) in parent(node)
        if !isdefault(e)
            @assert isnothing(e.weight.EnvR) && isnothing(e.weight.rightdata)
            @assert !isnothing(e.weight.EnvL) && !isnothing(e.weight.leftdata)
            e.weight.EnvR = EnvR
            e.weight.rightdata = deepcopy(rightdata)
            push!(tasks, e.weight)
            continue
        elseif length(child(p)) > 1
            e.weight.EnvR = EnvR
            e.weight.rightdata = deepcopy(rightdata)
            continue 
        end
        p.val.EnvR = deepcopy(EnvR)
        p.val.rightdata = deepcopy(rightdata)
        @assert !isnothing(p.val.EnvR)
        isnothing(p.val.EnvL) && push!(tasks, p)
    end
    node.val.EnvR = nothing
    node.val.rightdata = nothing
    return tasks
end

function _update!(node::DirectedNode, obj::T) where T <: Union{DenseMPS,DenseMPO}
    @assert !(isnothing(node.val.EnvL) && isnothing(node.val.EnvR))
    (!isnothing(node.val.EnvL) && !isnothing(node.val.EnvR)) && return _update!(node, obj, C2C())
    isnothing(node.val.EnvL) && return _update!(node, obj, R2L())
    isnothing(node.val.EnvR) && return _update!(node, obj, L2R())
end

function _update!(node::DirectedNode, obj::T, ::L2R) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    site = node.val.A.site
    site ≥ 1 && (node.val.EnvL = contract(obj[site],node.val.A,obj[site]',node.val.EnvL))
    if !node.val.isstring
        push!(node.val.leftdata["name"],node.val.A.name)
        push!(node.val.leftdata["site"],node.val.A.site)
    end
    return dispatch!(node,L2R())
end

function _update!(node::DirectedNode, obj::T, ::R2L) where T <: Union{DenseMPS{L},DenseMPO{L}} where L
    @assert isnothing(node.val.EnvL)
    @assert !isnothing(node.val.EnvR)
    site = node.val.A.site
    site ≤ L && (node.val.EnvR = contract(obj[site],node.val.A,obj[site]',node.val.EnvR))
    if !node.val.isstring
        push!(node.val.rightdata["name"],node.val.A.name)
        push!(node.val.rightdata["site"],node.val.A.site)
    end
    return dispatch!(node,R2L())
end


function _update!(weight::ObservableWeight, ::T) where T <: Union{DenseMPS,DenseMPO}
    ans = contract(weight.EnvL, weight.EnvR)
    names, sites = _lr_merge(weight.leftdata, weight.rightdata)
    default_weight!(weight)
    return (names, sites, ans)
end

function _update!(node::DirectedNode, obj::T, ::C2C) where T <: Union{DenseMPS,DenseMPO}
    site = node.val.A.site
    ans = contract(contract(obj[site], node.val.A, obj[site]', node.val.EnvL), node.val.EnvR)
    if !node.val.isstring
        push!(node.val.leftdata["name"],node.val.A.name)
        push!(node.val.leftdata["site"],node.val.A.site)
    end
    names, sites = _lr_merge(node.val.leftdata, node.val.rightdata)
    default_val!(node)
    return (names, sites, ans)
end