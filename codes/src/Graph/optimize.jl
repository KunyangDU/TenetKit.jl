
# ========================= 逐层优化 =========================
_lr_sig(n::DirectedNode) = "$(join(sort!([string(objectid(p)) for p in n.in]), ","))|$(n.val)"
_rl_sig(n::DirectedNode) = "$(join(sort!([string(objectid(c)) for c in n.out]), ","))|$(n.val)"

# 单层 merge
_lr_merge_at!(layers::Vector{Vector{DirectedNode}}, pos::Int) = (layers[pos] = _merge_by!(_lr_sig, layers[pos]))
_rl_merge_at!(layers::Vector{Vector{DirectedNode}}, pos::Int) = (layers[pos] = _merge_by!(_rl_sig, layers[pos]))

# L→R 单层 split：多 in 节点拆一个 parent 到已有单 in 节点
function _lr_split_at!(layers::Vector{Vector{DirectedNode}}, pos::Int)
    layer = layers[pos]
    for n in filter(x -> length(x.in) > 1 && !issentinel(x) &&
                        !isempty(x.out), layer)
        n in layer || continue
        for parent in copy(n.in)
            issentinel(parent) && continue
            for m in layer
                m === n && continue
                isempty(m.in) && isempty(m.out) && continue
                if isequal(m.val, n.val) && length(m.in) == 1 && m.in[1] === parent
                    _remove_edge!(parent, n)
                    for c in copy(n.out)
                        add_edge!(m, c)
                    end
                    return true
                end
            end
        end
    end
    return false
end

# R→L 单层 split：多 out 节点拆一个 child 到已有单 out 节点
function _rl_split_at!(layers::Vector{Vector{DirectedNode}}, pos::Int)
    layer = layers[pos]
    for n in filter(x -> length(x.out) > 1 && !issentinel(x) &&
                        !isempty(x.in), layer)
        n in layer || continue
        for child in copy(n.out)
            issentinel(child) && continue
            for m in layer
                m === n && continue
                isempty(m.in) && isempty(m.out) && continue
                if isequal(m.val, n.val) && length(m.out) == 1 && m.out[1] === child
                    _remove_edge!(n, child)
                    for p in copy(n.in)
                        add_edge!(p, m)
                    end
                    return true
                end
            end
        end
    end
    return false
end

# 逐层优化：merge → (split + remerge 循环至稳定)
function _lr_optimize!(layers::Vector{Vector{DirectedNode}})
    for pos in 1:length(layers)
        _lr_merge_at!(layers, pos)
        while _lr_split_at!(layers, pos)
            _lr_merge_at!(layers, pos)
        end
    end
end

function _rl_optimize!(layers::Vector{Vector{DirectedNode}})
    for pos in reverse(1:length(layers))
        _rl_merge_at!(layers, pos)
        while _rl_split_at!(layers, pos)
            _rl_merge_at!(layers, pos)
        end
    end
end

# ========================= 层提取 =========================
# 从 DAG 的 source 出发，逐层收集物理节点
function _extract_layers(dag::LayerDirectedAcyclicGraph)
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
    optimize!(dag::LayerDirectedAcyclicGraph)

对 LayerDirectedAcyclicGraph 运行 Moore 框架下的逐层 merge-split 优化，迭代至不动点。
"""
function optimize!(dag::LayerDirectedAcyclicGraph)
    layers = _extract_layers(dag)
    isempty(layers) && return dag

    # 迭代至不动点
    while true
        old = [length(l) for l in layers]
        _lr_optimize!(layers)
        _rl_optimize!(layers)
        [length(l) for l in layers] == old && break
    end

    for layer in layers
        filter!(n -> !(isempty(n.in) && isempty(n.out)), layer)
    end

    return dag
end
