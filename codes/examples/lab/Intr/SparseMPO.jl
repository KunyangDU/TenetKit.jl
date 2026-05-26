# SparseMPO v2.0: 从 InteractionGraph 的 DAG 构建稀疏 MPO
#
#   图有 L+2 层（L 物理层 + entry/exit 哨兵层），收缩 MPS-MPO-MPS'
#   本质是推环境：入口哨兵赋 isometry 初始环境，逐层缩并推进。
#   MPO 编码环境的映射关系。
#
# 数据结构:
#   SparseMPO{L}  — L 层稀疏 MPO
#     ts::Vector{SparseMPOTensor}    各位置的张量 (长度 L)
#     D ::NTuple{L,NTuple{3,Int64}}  D[i] = (dim_in, op_count, dim_out)
#
#   SparseMPOTensor{DL,D,DR} — 位置 i 的张量
#     DL = bond i-1→i 的左环境维数 = |A_{i-1}|
#     D  = 位置 i 的算符种类数     = |A_i|
#     DR = bond i→i+1 的右环境维数 = |A_{i+1}|
#
#     A::Vector{LocalOperator}    算符列表 (长度 D)
#     left::BondMap{1,DL,D}   左 bond 节点 ↔ 本层算符
#     right::BondMap{1,D,DR}  本层算符 ↔ 右 bond 节点
#
#   BondMap (见 Intr/map.jl):
#     从 BondMap 的 fwd/rev 可导出:
#       left.fwd[p]   = 左指标 p → 本层算符
#       right.rev[c]  = 右指标 c ← 本层算符
#       (left.rev[j], right.fwd[j]) = 算符 j 的左右连通
#     compose(R₁, R₂) 实现关系复合，用于多层穿透
#
#   左→右推进: EnvR_new[c] = Σ_{j∈right.rev[c]} (Σ_{p∈left.rev[j]} EnvL[p]) * A[j]
#   右→左推进: EnvL_new[p] = Σ_{j∈left.fwd[p]} A[j] * (Σ_{c∈right.fwd[j]} EnvR[c])

# ========================= 数据结构 =========================

mutable struct SparseMPOTensor{DL,D,DR}
    A::Vector{LocalOperator}
    left::BondMap{1,DL,D}    # 左 bond 节点 ↔ 本层算符
    right::BondMap{1,D,DR}   # 本层算符 ↔ 右 bond 节点
end

mutable struct SparseMPO{L}
    ts::Vector{SparseMPOTensor}
    D::NTuple{L,NTuple{3,Int64}}
end

function getindex(obj::SparseMPO{L}, i::Int64) where L
    1 <= i <= L || throw(BoundsError(obj, i))
    return obj.ts[i]
end

# ========================= 从 InteractionGraph 构建 SparseMPO =========================

# 解析入口哨兵：穿透到物理节点或出口哨兵
function _resolve_forward(node::InteractionGraphNode)
    if issentinel(node) && !isempty(node.out)
        result = InteractionGraphNode[]
        for c in node.out
            append!(result, _resolve_forward(c))
        end
        return result
    end
    return [node]   # 物理节点 或 出口哨兵
end

# 收集所有层（含哨兵层），返回 Vector{Vector{InteractionGraphNode}}
function _collect_layers(root::InteractionGraphNode)
    layers = Vector{InteractionGraphNode}[]
    cur = [root]
    push!(layers, cur)   # layer 1: 入口哨兵

    while true
        next_dict = Dict{Any,Int}()   # 去重 + 保序
        for n in cur
            for c in n.out
                for r in _resolve_forward(c)
                    get!(next_dict, r, length(next_dict) + 1)
                end
            end
        end
        isempty(next_dict) && break
        next_layer = sort!(collect(keys(next_dict)), by = x -> next_dict[x])
        push!(layers, next_layer)

        if all(x -> issentinel(x) && isempty(x.out), next_layer)
            break
        end
        cur = next_layer
    end
    return layers
end

# 从相邻层直接构建 BondMap{1}，不经过 Dict
function _build_bondmap(prev_layer::Vector, curr_layer::Vector)
    nA = length(prev_layer)
    nB = length(curr_layer)
    node2idx = Dict{Any,Int64}(n => Int64(i) for (i, n) in enumerate(curr_layer))
    return BondMap(nA, nB) do a
        p = prev_layer[a]
        idxs = Int64[]
        for c in p.out
            for r in _resolve_forward(c)
                ci = get(node2idx, r, nothing)
                ci !== nothing && push!(idxs, ci)
            end
        end
        return idxs
    end
end

# 从 BondMap{1} 构建 SparseMPOTensor，DL,D,DR 从 BondMap 类型参数推断
function _build_tensor(A, left::BondMap{1,DL,D}, right::BondMap{1,D,DR}) where {DL,D,DR}
    return SparseMPOTensor{DL,D,DR}(A, left, right)
end

"""
    build_sparse_mpo(ig::InteractionGraph{L}) -> SparseMPO{L}

从 InteractionGraph 的 DAG 构建 SparseMPO。图有 L+2 层（L 物理层 + 2 哨兵层）。
"""
function build_sparse_mpo(ig::InteractionGraph{L}) where L
    isnothing(ig.graph) && build_graph!(ig)
    root = ig.graph.node[1]

    layers = _collect_layers(root)
    @assert length(layers) == L + 2 "Expected $(L+2) layers, got $(length(layers))"

    # 构建全部 bond 的 BondMap{1}
    bond_maps = BondMap[]
    for b in 1:L+1
        push!(bond_maps, _build_bondmap(layers[b], layers[b+1]))
    end

    # 逐层构建 SparseMPOTensor
    ts_vec = SparseMPOTensor[]
    for i in 1:L
        A = LocalOperator[layers[i+1][j].val for j in 1:length(layers[i+1])]
        tensor = _build_tensor(A, bond_maps[i], bond_maps[i+1])
        push!(ts_vec, tensor)
    end

    # D[i] = (dim_in, op_count, dim_out) = (|left.fwd|, |A|, |right.rev|)
    D_tuple = ntuple(L) do i
        (Int64(length(ts_vec[i].left.fwd)), Int64(length(ts_vec[i].A)), Int64(length(ts_vec[i].right.rev)))
    end

    return SparseMPO{L}(ts_vec, D_tuple)
end
