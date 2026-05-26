# ========================= MPO 迭代器 =========================
# 从左至右逐格点遍历 DAG，每步输出稀疏 MPO 矩阵
#
# 映射关系：ind_{i-1} ── Op_i ── ind_{i+1}
#   row = parent_idx (ind_{i-1}), col = child_idx (ind_{i+1})
#   matrix element = child.val (该格点的算符)
#
# MPO_1 是 (1, N_1) 矩阵，MPO_L 是 (N_L, 1) 矩阵
# 整体构成 SparseMPO

abstract type AbstractGraphIterator end

mutable struct MPOIterator <: AbstractGraphIterator
    parent::Vector{InteractionGraphNode}   # 当前前沿节点 (ind_{i-1})
    done::Bool
end

# -- 哨兵解析：统一穿透入口/出口哨兵 --
# 入口哨兵（有子节点）→ 递归穿透到非哨兵节点
# 出口哨兵（无子节点）→ 保留为终端
# 普通节点            → 保留自身
function _resolve_sentinels(node::InteractionGraphNode)
    if issentinel(node) && !isempty(node.out)
        result = eltype(node)[]
        for c in node.out
            append!(result, _resolve_sentinels(c))
        end
        return result
    end
    return [node]
end

"""
    MPOIterator(root::InteractionGraphNode)

从 DAG 根节点构造迭代器。若入口非哨兵则包裹虚拟哨兵，保证第一 site 的
算符被正确输出。
"""
function MPOIterator(root::InteractionGraphNode)
    if !issentinel(root)
        virtual = sentinel(typeof(root.val))
        add_edge!(virtual, root)
        root = virtual
    end
    return MPOIterator([root], false)
end

"""
    next_mpo!(iter::MPOIterator) -> Union{Nothing, Tuple}

返回当前格点的稀疏 MPO 矩阵 `(rows, cols, ops)`：
- `rows[p]` — 父节点索引（从 1 开始）
- `cols[p]` — 子节点索引（从 1 开始）
- `ops[p]`  — 矩阵元算符

迭代结束时返回 `nothing`。
"""
function next_mpo!(iter::MPOIterator)
    iter.done && return nothing

    child_idx_map = Dict{Any, Int}()
    rows = Int[]
    cols = Int[]
    ops  = []

    for (p_idx, p) in enumerate(iter.parent)
        for c in p.out
            # 统一穿透入口哨兵，得到本层算符节点
            op_nodes = _resolve_sentinels(c)

            for op_node in op_nodes
                issentinel(op_node) && continue   # 出口哨兵：终端，无算符

                # 向前看一步：子节点经哨兵解析后若是出口哨兵 → 末级
                nc = isempty(op_node.out) ? nothing : first(_resolve_sentinels(first(op_node.out)))
                if nc !== nothing && issentinel(nc) && isempty(nc.out)
                    c_idx = get!(child_idx_map, nc, length(child_idx_map) + 1)
                    push!(rows, p_idx); push!(cols, c_idx); push!(ops, op_node.val)
                else
                    c_idx = get!(child_idx_map, op_node, length(child_idx_map) + 1)
                    push!(rows, p_idx); push!(cols, c_idx); push!(ops, op_node.val)
                end
            end
        end
    end

    if isempty(child_idx_map)
        iter.done = true
        return nothing
    end

    sorted = sort!(collect(child_idx_map), by = kv -> kv[2])
    iter.parent = [kv[1] for kv in sorted]

    return (rows, cols, ops)
end

# ========================= 构造 SparseMPO =========================

"""
    build_sparse_mpo(ig::InteractionGraph) -> SparseMPO

从 InteractionGraph 构建 SparseMPO。若尚未建图则先调用 `build_graph!`，
然后逐格点迭代生成稀疏 MPO 张量。D[i] = size(ts[i])。
"""
function build_sparse_mpo(ig::InteractionGraph)
    isnothing(ig.graph) && build_graph!(ig)
    root = ig.graph.node[1]

    iter = MPOIterator(root)
    tensors = SparseMPOTensor[]
    shapes  = NTuple{2,Int64}[]

    while true
        result = next_mpo!(iter)
        result === nothing && break
        rows, cols, ops = result
        N, M = maximum(rows), maximum(cols)

        m = Matrix{Union{Nothing, AbstractLocalOperator}}(nothing, N, M)
        for k in 1:length(rows)
            m[rows[k], cols[k]] = ops[k]
        end
        push!(tensors, SparseMPOTensor(m))
        push!(shapes, (N, M))
    end

    return SparseMPO(tensors, shapes)
end