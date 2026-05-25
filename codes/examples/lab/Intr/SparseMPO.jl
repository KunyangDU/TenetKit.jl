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
#     A::Vector{LocalOperator}              算符列表 (长度 D)
#     l2r::Vector{Vector{Int64}}            左环境→算符映射 (长度 DL): l2r[p] = 左指标 p 连接的本层算符
#     r2l::Vector{Vector{Int64}}            右环境→算符映射 (长度 DR): r2l[c] = 右指标 c 接收的本层算符
#     validind::NTuple{D,NTuple{2,Vector{Int64}}}  算符→环境映射: validind[j] = (左环境指标, 右环境指标)
#
#   关系: j ∈ l2r[p] ⇔ p ∈ validind[j][1];  j ∈ r2l[c] ⇔ c ∈ validind[j][2]
#
#   左→右推进: EnvR_new[c] = Σ_{j∈r2l[c]} (Σ_{p∈validind[j][1]} EnvL[p]) * A[j]
#   右→左推进: EnvL_new[p] = Σ_{j∈l2r[p]} A[j] * (Σ_{c∈validind[j][2]} EnvR[c])

# ========================= 数据结构 =========================

mutable struct SparseMPO{L}
    ts::Vector{SparseMPOTensor}
    D::NTuple{L,NTuple{3,Int64}}
end

mutable struct SparseMPOTensor{DL,D,DR}
    A::Vector{LocalOperator}
    l2r::Vector{Vector{Int64}}
    r2l::Vector{Vector{Int64}}
    validind::NTuple{D,NTuple{2,Vector{Int64}}}
end

function getindex(obj::SparseMPO{L}, i::Int64) where L
    1 <= i <= L || throw(BoundsError(obj, i))
    return obj.ts[i]
end

# ========================= 编译期 Val 派发 =========================
# SparseMPOTensor{DL,D,DR} 的类型参数必须在编译期已知，
# 但层大小是运行时从图计算得到的。用 @generated 生成派发树。

struct Val3{DL,D,DR} end

const _SPARSE_MPO_MAX_BOND = 32

@generated function _dispatch_val3(dl::Int, d::Int, dr::Int, ::Val{Max}) where Max
    # 嵌套 if-else 树: dl → d → dr，从 1 到 Max
    # 迭代 Max:-1:1 让常见小值在最外层 (dl==1 先检查)
    outer = :(error("Bond dimension ($dl,$d,$dr) exceeds _SPARSE_MPO_MAX_BOND=$Max; increase _SPARSE_MPO_MAX_BOND"))
    for DL in Max:-1:1
        mid = :(error("Bond dimension ($dl,$d,$dr) exceeds _SPARSE_MPO_MAX_BOND=$Max"))
        for D in Max:-1:1
            inner = :(error("Bond dimension ($dl,$d,$dr) exceeds _SPARSE_MPO_MAX_BOND=$Max"))
            for DR in Max:-1:1
                val = :(Val3{$DL,$D,$DR}())
                inner = :(if dr == $DR; $val; else; $inner; end)
            end
            mid = :(if d == $D; $inner; else; $mid; end)
        end
        outer = :(if dl == $DL; $mid; else; $outer; end)
    end
    return outer
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

# 构建单层 l2r: 从 prev_layer 到 curr_layer 的映射
function _build_l2r(prev_layer::Vector, curr_layer::Vector)
    mapping = Dict{Int64,Vector{Int64}}()
    for (p_idx, p) in enumerate(prev_layer)
        child_indices = Int64[]
        for c in p.out
            for r in _resolve_forward(c)
                c_idx = findfirst(x -> x === r, curr_layer)
                if c_idx !== nothing
                    push!(child_indices, Int64(c_idx))
                end
            end
        end
        if !isempty(child_indices)
            mapping[Int64(p_idx)] = child_indices
        end
    end
    return mapping
end

# 构建单层 SparseMPOTensor (编译期 DL,D,DR 已知)
function _build_tensor(layers_curr,
                       fwd_prev::Dict{Int64,Vector{Int64}},
                       fwd_next::Dict{Int64,Vector{Int64}},
                       ::Val3{DL,D,DR}) where {DL,D,DR}
    # A: 该位置全部算符
    A = LocalOperator[layers_curr[j].val for j in 1:D]

    # Invert fwd_prev → 每个算符的左环境指标
    inv_prev = Dict{Int64,Vector{Int64}}()
    for (p, c_indices) in fwd_prev
        for c in c_indices
            vec = get!(inv_prev, c, Int64[])
            push!(vec, p)
        end
    end

    # Invert fwd_next → 每个右环境指标接收的算符
    inv_next = Dict{Int64,Vector{Int64}}()
    for (j, c_indices) in fwd_next
        for c in c_indices
            vec = get!(inv_next, c, Int64[])
            push!(vec, j)
        end
    end

    # l2r[p]: 左环境指标 p 连接的本层算符列表
    l2r = Vector{Int64}[sort!(get(fwd_prev, Int64(p), Int64[])) for p in 1:DL]

    # r2l[c]: 右环境指标 c 接收的本层算符列表
    r2l = Vector{Int64}[sort!(get(inv_next, Int64(c), Int64[])) for c in 1:DR]

    # validind[j]: (该算符的左环境指标, 该算符的右环境指标)
    validind = ntuple(Val(D)) do j
        lefts  = sort!(get(inv_prev, Int64(j), Int64[]))
        rights = sort!(get(fwd_next, Int64(j), Int64[]))
        (lefts, rights)
    end

    return SparseMPOTensor{DL,D,DR}(A, l2r, r2l, validind)
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

    # 构建全部 bond 的 l2r 映射
    l2r_maps = Dict{Int64,Vector{Int64}}[]
    for b in 1:L+1
        push!(l2r_maps, _build_l2r(layers[b], layers[b+1]))
    end

    # 逐层构建 SparseMPOTensor
    ts_vec = SparseMPOTensor[]
    for i in 1:L
        dl = length(layers[i])
        d  = length(layers[i+1])
        dr = length(layers[i+2])

        tensor = _build_tensor(layers[i+1],
                                l2r_maps[i], l2r_maps[i+1],
                                _dispatch_val3(dl, d, dr, Val(_SPARSE_MPO_MAX_BOND)))
        push!(ts_vec, tensor)
    end

    # D[i] = (dim_in, op_count, dim_out) = (|l2r|, |A|, |r2l|)
    D_tuple = ntuple(L) do i
        (Int64(length(ts_vec[i].l2r)), Int64(length(ts_vec[i].A)), Int64(length(ts_vec[i].r2l)))
    end

    return SparseMPO{L}(ts_vec, D_tuple)
end
