# ============================================================
# IntrNode — DAG 数据结构 + 最优图构建
#
#   1. 初始独立链，共享 entry/exit 哨兵
#   2. L→R 逐层优化: merge (in_set,val) → split 多 in → remerge
#   3. R→L 逐层优化: merge (out_set,val) → split 多 out → remerge
#   4. 迭代至不动点
# ============================================================

# ========================= 数据结构 =========================
mutable struct InteractionGraphNode{T}
    val::Union{T, Nothing}
    in ::Vector{InteractionGraphNode}
    out::Vector{InteractionGraphNode}
end

InteractionGraphNode(val::T) where T = InteractionGraphNode{T}(val, Vector{InteractionGraphNode}(), Vector{InteractionGraphNode}())

sentinel(::Type{T}) where T = InteractionGraphNode{T}(nothing, Vector{InteractionGraphNode}(), Vector{InteractionGraphNode}())
issentinel(n::InteractionGraphNode) = n.val === nothing

function add_edge!(from::InteractionGraphNode, to::InteractionGraphNode)
    to in from.out && return
    push!(from.out, to)
    push!(to.in, from)
end

# ========================= 图操作 =========================
function _remove_edge!(from::InteractionGraphNode, to::InteractionGraphNode)
    idx = findfirst(x -> x === to, from.out)
    idx !== nothing && deleteat!(from.out, idx)
    idx = findfirst(x -> x === from, to.in)
    idx !== nothing && deleteat!(to.in, idx)
end

function _merge_into!(keeper::InteractionGraphNode, other::InteractionGraphNode)
    for c in copy(other.out)
        _remove_edge!(other, c)
        add_edge!(keeper, c)
    end
    for p in copy(other.in)
        _remove_edge!(p, other)
        add_edge!(p, keeper)
    end
    empty!(other.out)
    empty!(other.in)
end

function _merge_by!(sig_fn::Function, layer::Vector{InteractionGraphNode})
    groups = Dict{String, Vector{InteractionGraphNode}}()
    for n in layer
        isempty(n.in) && isempty(n.out) && continue
        sig = sig_fn(n)
        vec = get!(groups, sig) do; InteractionGraphNode[]; end
        push!(vec, n)
    end
    new_layer = InteractionGraphNode[]
    for (_, group) in groups
        keeper = group[1]
        for n in group[2:end]
            _merge_into!(keeper, n)
        end
        push!(new_layer, keeper)
    end
    return new_layer
end

# ========================= 逐层优化 =========================
_lr_sig(n) = "$(join(sort!([string(objectid(p)) for p in n.in]), ","))|$(n.val)"
_rl_sig(n) = "$(join(sort!([string(objectid(c)) for c in n.out]), ","))|$(n.val)"

# 单层 merge
_lr_merge_at!(layers, pos) = (layers[pos] = _merge_by!(_lr_sig, layers[pos]))
_rl_merge_at!(layers, pos) = (layers[pos] = _merge_by!(_rl_sig, layers[pos]))

# L→R 单层 split：多 in 节点拆一个 parent 到已有单 in 节点
function _lr_split_at!(layers, pos)
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
function _rl_split_at!(layers, pos)
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
function _lr_optimize!(layers)
    for pos in 1:length(layers)
        _lr_merge_at!(layers, pos)
        while _lr_split_at!(layers, pos)
            _lr_merge_at!(layers, pos)
        end
    end
end

function _rl_optimize!(layers)
    for pos in reverse(1:length(layers))
        _rl_merge_at!(layers, pos)
        while _rl_split_at!(layers, pos)
            _rl_merge_at!(layers, pos)
        end
    end
end

# ========================= 主函数 =========================
"""
    build_optimal_dag(seqs)

从序列集构建最优 DAG（Moore 框架内最小节点数）。
返回 (entry, layers)。
"""
function build_optimal_dag(seqs::Vector{Vector{T}}) where T
    isempty(seqs) && return (nothing, Vector{InteractionGraphNode}[])
    L = length(seqs[1])

    # 初始独立链，共享 entry/exit 哨兵
    entry = sentinel(T)
    exit_s = sentinel(T)
    layers = [InteractionGraphNode[] for _ in 1:L]

    for seq in seqs
        prev = entry
        for (pos, val) in enumerate(seq)
            node = InteractionGraphNode(val)
            add_edge!(prev, node)
            push!(layers[pos], node)
            prev = node
        end
        add_edge!(prev, exit_s)
    end

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

    return entry, layers
end

# ========================= 辅助函数 =========================
function layer_counts(root::InteractionGraphNode)
    cur = issentinel(root) ? collect(root.out) : [root]
    visited = Set{Any}(cur)
    counts = Int[]
    while !isempty(cur)
        real = count(n -> !issentinel(n), cur)
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
