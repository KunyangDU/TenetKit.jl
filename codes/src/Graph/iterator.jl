# ========================= LayerIterator =========================
# 从 LayerDirectedAcyclicGraph 的 source sentinel 出发，逐层推进至 sink sentinel
# 每次迭代 yield (left::LayerMap, vals::Vector, right::LayerMap)

struct LayerIterator{G<:LayerDirectedAcyclicGraph}
    dag::G
end

function Base.iterate(iter::LayerIterator)
    entry = [iter.dag.source[1]]
    cur = _next_nodes(entry)
    nxt = _next_nodes(cur)
    return ((LayerMap(entry, cur), _layervals(cur), LayerMap(cur, nxt)), (cur, nxt))
end

function Base.iterate(::LayerIterator, state::Tuple{Vector,Vector})
    prev, cur = state
    nxt = _next_nodes(cur)
    isempty(nxt) && return nothing
    left = LayerMap(prev, cur)
    right = LayerMap(cur, nxt)
    vals = _layervals(cur)
    return ((left, vals, right), (cur, nxt))
end

Base.IteratorSize(::Type{<:LayerIterator}) = Base.SizeUnknown()

# -- 收集当前层节点的 val --
_layervals(cur::Vector{T}) where T = T[n.val for n in cur]
