# ============================================================
# DirectedNode — 有向无环图节点 + 图操作
#
#   1. 初始独立链，共享 entry/exit 哨兵
#   2. L→R 逐层优化: merge (in_set,val) → split 多 in → remerge
#   3. R→L 逐层优化: merge (out_set,val) → split 多 out → remerge
#   4. 迭代至不动点
# ============================================================

# ========================= 数据结构 =========================
mutable struct DirectedNode{T}
    val::Union{T, Nothing}
    in_edges::Vector
    out_edges::Vector
end

DirectedNode(val::T) where T = DirectedNode{T}(val, Vector(), Vector())

sentinel(::Type{T}) where T = DirectedNode{T}(nothing, Vector(), Vector())
issentinel(n::DirectedNode) = n.val === nothing


# ========================= 图遍历 =========================
# -- 哨兵解析：穿透非出口哨兵，保留物理节点和出口哨兵 --
function _resolve_forward(node::DirectedNode)
    if issentinel(node) && !isempty(node.out_edges)
        result = DirectedNode[]
        for e in node.out_edges
            append!(result, _resolve_forward(e.to))
        end
        return result
    end
    return [node]
end

# -- 从当前节点集推进到下一层节点（去重 + 保序）--
function _next_nodes(cur::Vector)
    node_dict = Dict{Any,Int}()
    for n in cur
        for e in n.out_edges
            for r in _resolve_forward(e.to)
                get!(node_dict, r, length(node_dict) + 1)
            end
        end
    end
    isempty(node_dict) && return DirectedNode[]
    result = sort!(collect(keys(node_dict)), by = x -> node_dict[x])
    return convert(Vector{DirectedNode}, result)
end

child(node::DirectedNode) = map(x -> (x,x.to), node.out_edges)
parent(node::DirectedNode) = map(x -> (x,x.from), node.in_edges)

function Base.show(io::IO, node::DirectedNode)
    print(io, "$(typeof(node))($(length(node.in_edges)) → $(node.val) → $(length(node.out_edges)))")
    # print(io, "$(typeof(node))($(tuple(map(x -> x.from.val,node.in_edges)...)) → $(node.val) → $(tuple(map(x -> x.to.val,node.out_edges)...)))")
end