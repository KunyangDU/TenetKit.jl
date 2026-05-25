# ============================================================
# IntrNode — DAG 数据结构与图优化
# ============================================================

# ========================= 数据结构 =========================
mutable struct InteractionGraphNode{T}
    val::Union{T, Nothing}
    in::Vector{InteractionGraphNode}
    out::Vector{InteractionGraphNode}
    is_op::Bool
end

InteractionGraphNode(val::T) where T = InteractionGraphNode{T}(val, Vector{InteractionGraphNode}(), Vector{InteractionGraphNode}(), false)
sentinel(::Type{T}) where T = InteractionGraphNode{T}(nothing, Vector{InteractionGraphNode}(), Vector{InteractionGraphNode}(), false)

issentinel(n::InteractionGraphNode) = n.val === nothing
isleaf(n::InteractionGraphNode)     = length(n.out) == 0
isfork(n::InteractionGraphNode)     = length(n.out) > 1
isjoin(n::InteractionGraphNode)     = length(n.in) > 1

function add_edge!(from::InteractionGraphNode, to::InteractionGraphNode)
    to in from.out && return
    push!(from.out, to)
    push!(to.in, from)
end

_show_val(val) = sprint(show, val)

# ========================= DAG 全局最小化 =========================
# 迭代精炼等价类：两个节点等价 ⇔ 值相同 ∧ 所有出边指向等价节点

function _eff_children(node::InteractionGraphNode)
    result = eltype(node.out)[]
    for c in node.out
        if issentinel(c)
            if isempty(c.out)
                push!(result, c)
            else
                append!(result, _eff_children(c))
            end
        else
            push!(result, c)
        end
    end
    return result
end

function _collect_all!(node::InteractionGraphNode, result, seen)
    node in seen && return
    push!(seen, node)
    push!(result, node)
    for child in node.out
        _collect_all!(child, result, seen)
    end
end

"""
    minimize!(root::InteractionGraphNode)

全局最小化 DAG，合并等价节点，原地修改。
出口哨兵视为统一的"接受类"，保证左右对称。
"""
function minimize!(root::InteractionGraphNode)
    all_nodes = InteractionGraphNode[]
    _collect_all!(root, all_nodes, Set{Any}())

    exit_sentinels = InteractionGraphNode[]
    class = Dict{InteractionGraphNode, Int}()

    val_order = Any[]
    val_ids   = Int[]

    for n in all_nodes
        if issentinel(n)
            isempty(n.out) && push!(exit_sentinels, n)
            continue
        end
        idx = findfirst(x -> isequal(x, n.val), val_order)
        if idx === nothing
            push!(val_order, n.val)
            push!(val_ids, length(val_ids) + 1)
            class[n] = val_ids[end]
        else
            class[n] = val_ids[idx]
        end
    end

    exit_cid = length(val_ids) + 1
    for s in exit_sentinels
        class[s] = exit_cid
    end

    while true
        sig2id = Dict{String, Int}()
        new_class = Dict{InteractionGraphNode, Int}()
        next_id = 1

        for n in all_nodes
            issentinel(n) && !isempty(n.out) && continue
            eff = _eff_children(n)
            valid_children = [c for c in eff if !issentinel(c) || isempty(c.out)]
            child_ids = Int[class[c] for c in valid_children]
            sort!(child_ids)
            sig = _show_val(issentinel(n) ? nothing : n.val) * "|" * join(child_ids, ",")

            id = get!(sig2id, sig) do
                id = next_id; next_id += 1; id
            end
            new_class[n] = id
        end

        stable = true
        for n in all_nodes
            issentinel(n) && !isempty(n.out) && continue
            if new_class[n] != get(class, n, 0)
                stable = false; break
            end
        end
        stable && break
        class = new_class
    end

    rep = Dict{Int, InteractionGraphNode}()
    for n in all_nodes
        issentinel(n) && !isempty(n.out) && continue
        get!(rep, class[n], n)
    end

    new_edges = Dict{InteractionGraphNode, Vector{InteractionGraphNode}}()
    for n in all_nodes
        children = InteractionGraphNode[]
        for c in _eff_children(n)
            if issentinel(c) && !isempty(c.out)
                c in children || push!(children, c)
            elseif issentinel(c)
                r = rep[class[c]]
                r in children || push!(children, r)
            else
                r = rep[class[c]]
                r in children || push!(children, r)
            end
        end
        new_edges[n] = children
    end

    for n in all_nodes
        empty!(n.out); empty!(n.in)
    end
    for (n, children) in new_edges
        for c in children
            add_edge!(n, c)
        end
    end

    root
end

# ========================= 右上下文态合并 =========================
# 将"子节点相同、值不同"的 Moore 节点通过 state sentinel 共享后继
# 仅在结果保持/改善对称性时执行合并

_is_state_sentinel(n::InteractionGraphNode) = issentinel(n) && !isempty(n.out) && !isempty(n.in)

"""
    merge_same_future!(root::InteractionGraphNode)

在 minimize! 之后调用。识别具有相同有效子节点但不同值的节点，
通过 state sentinel 实现后继共享。仅在保持/改善对称性时执行。
"""
function merge_same_future!(root::InteractionGraphNode)
    # 层收集
    layers = Vector{InteractionGraphNode}[]
    if issentinel(root)
        cur = collect(root.out)
    else
        cur = [root]
    end
    visited = Set{Any}(cur)
    while !isempty(cur)
        real_nodes = filter(n -> !issentinel(n), cur)
        if !isempty(real_nodes)
            push!(layers, real_nodes)
        end
        nxt = InteractionGraphNode[]
        for n in cur
            for c in n.out
                c in visited && continue
                push!(visited, c); push!(nxt, c)
            end
        end
        cur = nxt
    end

    # 第一遍：识别所有合并组（不修改图）
    state_sentinels = Set{InteractionGraphNode}()

    function _eff_child_opaque(node)
        result = eltype(node.out)[]
        for c in node.out
            if c in state_sentinels
                push!(result, c)
            elseif issentinel(c)
                if isempty(c.out)
                    push!(result, c)
                else
                    append!(result, _eff_child_opaque(c))
                end
            else
                push!(result, c)
            end
        end
        return result
    end

    merge_plan = Tuple{Int, Vector{InteractionGraphNode}}[]

    for (li, layer) in enumerate(reverse(layers))
        real_li = length(layers) - li + 1
        groups = Dict{String, Vector{InteractionGraphNode}}()
        for n in layer
            eff = _eff_child_opaque(n)
            sig = join(sort!([string(objectid(c)) for c in eff]), ",")
            vec = get!(groups, sig) do; InteractionGraphNode[]; end
            push!(vec, n)
        end

        for (sig, group) in groups
            length(group) <= 1 && continue
            all(n -> isequal(n.val, group[1].val), group) && continue
            all(n -> all(c -> issentinel(c) && isempty(c.out), _eff_child_opaque(n)), group) && continue
            parents_seen = Set{InteractionGraphNode}()
            has_shared_parent = false
            for n in group
                for pred in n.in
                    pred in parents_seen && (has_shared_parent = true; break)
                    push!(parents_seen, pred)
                end
                has_shared_parent && break
            end
            has_shared_parent && continue

            push!(merge_plan, (real_li, group))
            T = typeof(group[1].val)
            dummy_state = sentinel(T)
            push!(state_sentinels, dummy_state)
        end
    end

    isempty(merge_plan) && return root

    # 对称性检查
    predicted = copy(_layer_counts(root))
    for (li, group) in merge_plan
        predicted[li] = predicted[li] - length(group) + 1
    end

    asymmetry_before = sum(abs(_layer_counts(root)[i] - _layer_counts(root)[end-i+1]) for i in 1:div(length(layers),2))
    asymmetry_after  = sum(abs(predicted[i] - predicted[end-i+1]) for i in 1:div(length(predicted),2))

    asymmetry_after > asymmetry_before && return root

    # 第二遍：执行合并
    state_sentinels = Set{InteractionGraphNode}()

    function _eff_child_opaque2(node)
        result = eltype(node.out)[]
        for c in node.out
            if c in state_sentinels
                push!(result, c)
            elseif issentinel(c)
                if isempty(c.out)
                    push!(result, c)
                else
                    append!(result, _eff_child_opaque2(c))
                end
            else
                push!(result, c)
            end
        end
        return result
    end

    for (li, group) in merge_plan
        valid = InteractionGraphNode[]
        for n in group
            isempty(n.in) && isempty(n.out) && continue
            push!(valid, n)
        end
        length(valid) <= 1 && continue

        T = typeof(valid[1].val)
        state = sentinel(T)
        push!(state_sentinels, state)

        all_children = InteractionGraphNode[]
        for n in valid
            for c in n.out
                c in all_children || push!(all_children, c)
            end
        end
        for c in all_children
            add_edge!(state, c)
        end

        for n in valid
            for pred in copy(n.in)
                idx = findfirst(x -> x === n, pred.out)
                idx !== nothing && deleteat!(pred.out, idx)
                add_edge!(pred, state)
            end
            for c in copy(n.out)
                idx = findfirst(x -> x === n, c.in)
                idx !== nothing && deleteat!(c.in, idx)
            end
            empty!(n.out)
            empty!(n.in)
        end
    end

    return root
end

# ========================= 辅助函数 =========================

function _count_valued_nodes(node::InteractionGraphNode, visited=Set{Any}())
    node in visited && return 0
    push!(visited, node)
    node.is_op && return 0
    cnt = issentinel(node) ? 0 : 1
    for child in node.out
        cnt += _count_valued_nodes(child, visited)
    end
    return cnt
end

function _layer_counts(root::InteractionGraphNode)
    if issentinel(root)
        cur = collect(root.out)
    else
        cur = [root]
    end
    visited = Set{Any}(cur)
    counts = Int[]
    while !isempty(cur)
        real = count(n -> (!issentinel(n) && !n.is_op) || _is_state_sentinel(n), cur)
        if real > 0; push!(counts, real); end
        nxt = InteractionGraphNode[]
        for n in cur
            for c in n.out
                c in visited && continue
                push!(visited, c); push!(nxt, c)
            end
        end
        cur = nxt
    end
    return counts
end
