
# ========================= 逐层优化 =========================
_lr_sig(n::DirectedNode) = "$(join(sort!([string(objectid(e.from)) for e in n.in_edges]), ","))|$(n.val)"
_rl_sig(n::DirectedNode) = "$(join(sort!([string(objectid(e.to)) for e in n.out_edges]), ","))|$(n.val)"

# 单层 merge
_lr_merge_at!(layers::Vector{Vector{DirectedNode}}, pos::Int) = (layers[pos] = _merge_by!(_lr_sig, layers[pos], L2R()))
_rl_merge_at!(layers::Vector{Vector{DirectedNode}}, pos::Int) = (layers[pos] = _merge_by!(_rl_sig, layers[pos], R2L()))

# L→R 单层 split：多 in 节点每个 parent 拆到新建节点
function _lr_split_at!(layers::Vector{Vector{DirectedNode}}, pos::Int)
    layer = layers[pos]
    changed = false
    for n in filter(x -> length(x.in_edges) > 1 && !issentinel(x) &&
                        !isempty(x.out_edges), layer)
        n in layer || continue
        for parent_edge in copy(n.in_edges)
            parent = parent_edge.from
            issentinel(parent) && continue
            length(n.in_edges) <= 1 && break
            m = DirectedNode(n.val)
            push!(layer, m)
            _remove_edge!(parent, n)
            add_edge!(parent, m; weight = parent_edge.weight)
            for out_e in copy(n.out_edges)
                add_edge!(m, out_e.to; weight = out_e.weight)
            end
            changed = true
        end
    end
    return changed
end

# R→L 单层 split：多 out 节点每个 child 拆到新建节点
function _rl_split_at!(layers::Vector{Vector{DirectedNode}}, pos::Int)
    layer = layers[pos]
    changed = false
    for n in filter(x -> length(x.out_edges) > 1 && !issentinel(x) &&
                        !isempty(x.in_edges), layer)
        n in layer || continue
        for child_edge in copy(n.out_edges)
            child = child_edge.to
            issentinel(child) && continue
            length(n.out_edges) <= 1 && break
            m = DirectedNode(n.val)
            push!(layer, m)
            _remove_edge!(n, child)
            add_edge!(m, child; weight = child_edge.weight)
            for in_e in copy(n.in_edges)
                add_edge!(in_e.from, m; weight = in_e.weight)
            end
            changed = true
        end
    end
    return changed
end

# 逐层优化：split 拆开 (保证 merge 时 push 安全) → merge 推 weight → 反向 merge 合回去
function _lr_optimize!(layers::Vector{Vector{DirectedNode}}, isrev::Bool)
    for pos in 1:length(layers)
        _lr_split_at!(layers, pos)
        _lr_merge_at!(layers, pos)
        isrev && _rl_merge_at!(layers, pos)   # 合回被 in_set 拆开的节点
    end
end

function _rl_optimize!(layers::Vector{Vector{DirectedNode}}, isrev::Bool)
    for pos in reverse(1:length(layers))
        _rl_split_at!(layers, pos)
        _rl_merge_at!(layers, pos)
        isrev && _lr_merge_at!(layers, pos)   # 合回被 out_set 拆开的节点
    end
end

function _center_optimize!(layers::Vector{Vector{DirectedNode}}, isrev::Bool)
    L′ = div(length(layers),2)
    for pos in 1:L′
        _lr_split_at!(layers, pos)
        _lr_merge_at!(layers, pos)
        isrev && _rl_merge_at!(layers, pos)   # 合回被 in_set 拆开的节点
    end
    for pos in reverse(L′+1:length(layers))
        _rl_split_at!(layers, pos)
        _rl_merge_at!(layers, pos)
        isrev && _lr_merge_at!(layers, pos)   # 合回被 out_set 拆开的节点
    end
end

# ========================= 层提取 =========================
# 从 DAG 的 source 出发，逐层收集物理节点
function _extract_layers(dag::DirectedAcyclicGraph)
    layers = Vector{DirectedNode}[]
    cur = [dag.source[1]]
    while true
        nxt = _next_nodes(cur)
        real = filter(!issentinel, nxt)
        isempty(real) && break
        push!(layers, real)
        cur = real
    end
    return layers
end

# ========================= 优化入口 =========================
"""
    optimize!(dag::DirectedAcyclicGraph)

对 DirectedAcyclicGraph 运行 Moore 框架下的逐层 merge-split 优化，迭代至不动点。
"""
function optimize!(dag::DirectedAcyclicGraph;N::Int64 = 20)
    layers = _extract_layers(dag)
    isempty(layers) && return dag
    # 迭代至不动点
    for i in 1:N
        old = [length(l) for l in layers]
        _lr_optimize!(layers, i ≠ 1)
        _rl_optimize!(layers, true)
        _center_optimize!(layers,true)
        [length(l) for l in layers] == old && break
    end

    for layer in layers
        filter!(n -> !(isempty(n.in_edges) && isempty(n.out_edges)), layer)
    end

    return dag
end
