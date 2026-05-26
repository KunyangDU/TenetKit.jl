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
    in ::Vector{DirectedNode}
    out::Vector{DirectedNode}
end

DirectedNode(val::T) where T = DirectedNode{T}(val, Vector{DirectedNode}(), Vector{DirectedNode}())

sentinel(::Type{T}) where T = DirectedNode{T}(nothing, Vector{DirectedNode}(), Vector{DirectedNode}())
issentinel(n::DirectedNode) = n.val === nothing

function add_edge!(from::DirectedNode, to::DirectedNode)
    to in from.out && return
    push!(from.out, to)
    push!(to.in, from)
end

# ========================= 图遍历 =========================
# -- 哨兵解析：穿透非出口哨兵，保留物理节点和出口哨兵 --
function _resolve_forward(node::DirectedNode)
    if issentinel(node) && !isempty(node.out)
        result = DirectedNode[]
        for c in node.out
            append!(result, _resolve_forward(c))
        end
        return result
    end
    return [node]
end

# -- 从当前节点集推进到下一层节点（去重 + 保序）--
function _next_nodes(cur::Vector)
    node_dict = Dict{Any,Int}()
    for n in cur
        for c in n.out
            for r in _resolve_forward(c)
                get!(node_dict, r, length(node_dict) + 1)
            end
        end
    end
    isempty(node_dict) && return DirectedNode[]
    return sort!(collect(keys(node_dict)), by = x -> node_dict[x])
end

# ========================= 图操作 =========================
function _remove_edge!(from::DirectedNode, to::DirectedNode)
    idx = findfirst(x -> x === to, from.out)
    idx !== nothing && deleteat!(from.out, idx)
    idx = findfirst(x -> x === from, to.in)
    idx !== nothing && deleteat!(to.in, idx)
end

function _merge_into!(keeper::DirectedNode, other::DirectedNode)
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

function _merge_by!(sig_fn::Function, layer::Vector{DirectedNode})
    groups = Dict{String, Vector{DirectedNode}}()
    for n in layer
        isempty(n.in) && isempty(n.out) && continue
        sig = sig_fn(n)
        vec = get!(groups, sig) do; DirectedNode[]; end
        push!(vec, n)
    end
    new_layer = DirectedNode[]
    for (_, group) in groups
        keeper = group[1]
        for n in group[2:end]
            _merge_into!(keeper, n)
        end
        push!(new_layer, keeper)
    end
    return new_layer
end
