# ========================= SparseMPO 构造 =========================
# 从 InteractionGraph 的 DAG 逐层推进构建 SparseMPO
#
#   entry -> [layer_1] -> [layer_2] -> ... -> [layer_L] -> exit
#
# 每步：从当前前沿出发，解析下一层节点，建立 LayerMap，再推进到下一层。

"""
    AutomataSparseMPO(ig::InteractionGraph{L}) -> SparseMPO{L}

从 InteractionGraph 的 DAG 逐层构建 SparseMPO。
"""
function AutomataSparseMPO(ig::InteractionGraph{L}) where L
    initialize!(ig)
    ts_vec = SparseMPOTensor[]
    prev_right = nothing
    for (left, A, right) in LayerIterator(ig.graph)
        l = isnothing(prev_right) ? left : prev_right
        push!(ts_vec, SparseMPOTensor(A, l, right))
        prev_right = right
    end
    D_tuple = ntuple(L) do i
        (Int64(length(ts_vec[i].left.fwd)), Int64(length(ts_vec[i].A)), Int64(length(ts_vec[i].right.rev)))
    end
    return SparseMPO(ts_vec, D_tuple)
end
