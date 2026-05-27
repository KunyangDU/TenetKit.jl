# ========================= 图操作 =========================


# -- weight 继承：仅当 bond weight 冲突时才推系数 --
# 合并时 other 的共享边会被删除，若 keeper 同位置 weight 相同则直接覆盖，无需推；
# 仅当 weight 不同 (冲突) 时才将各自 weight 推到非共享边，共享边统一重置为 1。
function _merge_inherit_weight!(keeper::DirectedNode, other::DirectedNode, ::Type{Float64}, ::L2R)
    # 无冲突：两节点入边 weight 相同，keeper 入边覆盖 other 路径
    # keeper.in_edges[1].weight ≈ other.in_edges[1].weight && return
    # 有冲突：各自入边 weight 推到各自出边，入边统一重置为 1
    for n in (keeper, other)
        @assert length(n.in_edges) == 1
        for in_e in n.in_edges
            w = in_e.weight
            length(n.out_edges) > 1 && (@assert in_e.weight == 1)
            # in_e.weight ≠ 1 && (@assert length(n.out_edges) == 1)
            for out_e in n.out_edges
                @assert 1 in (in_e.weight,out_e.weight)
                out_e.weight *= w
            end
            in_e.weight = 1.0
        end
    end
end

function _merge_inherit_weight!(keeper::DirectedNode, other::DirectedNode, ::Type{Float64}, ::R2L)
    # 无冲突：两节点出边 weight 相同，keeper 出边覆盖 other 路径
    # keeper.out_edges[1].weight ≈ other.out_edges[1].weight && return
    # 有冲突：各自出边 weight 推到各自入边，出边统一重置为 1
    for n in (keeper, other)
        @assert length(n.out_edges) == 1
        for out_e in n.out_edges
            w = out_e.weight
            length(n.in_edges) > 1 && (@assert out_e.weight == 1)
            # out_e.weight ≠ 1 && (@assert length(n.in_edges) == 1)
            for in_e in n.in_edges
                @assert 1 in (in_e.weight,out_e.weight)
                in_e.weight *= w
            end
            out_e.weight = 1.0
        end
    end
end

function _merge_into!(keeper::DirectedNode, other::DirectedNode, direction = L2R())
    _merge_inherit_weight!(keeper, other, Float64, direction)
    if direction == L2R()
        # L2R: 同 in-set → other 入边可删 (keeper 已有); 不同 out-set → other 出边并入 keeper
        for e in copy(other.out_edges)
            _remove_edge!(other, e.to)
            add_edge!(keeper, e.to; weight = e.weight)
        end
        for e in copy(other.in_edges)
            _remove_edge!(e.from, other)
        end
    else  # R2L: 对称 — 同 out-set → other 出边可删; 不同 in-set → other 入边并入 keeper
        for e in copy(other.out_edges)
            _remove_edge!(other, e.to)
        end
        for e in copy(other.in_edges)
            _remove_edge!(e.from, other)
            add_edge!(e.from, keeper; weight = e.weight)
        end
    end
    empty!(other.out_edges)
    empty!(other.in_edges)
end

function _merge_by!(sig_fn::Function, layer::Vector{<:DirectedNode}, direction = L2R())
    mergeable = filter(layer) do n
        isempty(n.in_edges) && isempty(n.out_edges) && return false
        if direction == L2R()
            length(n.in_edges) == 1
        else  # :rl
            length(n.out_edges) == 1
        end
    end
    groups = Dict{String, Vector{DirectedNode}}()
    for n in mergeable
        sig = sig_fn(n)
        vec = get!(groups, sig) do; DirectedNode[]; end
        push!(vec, n)
    end
    new_layer = DirectedNode[]
    for (_, group) in groups
        keeper = group[1]
        for n in group[2:end]
            _merge_into!(keeper, n, direction)
        end
        push!(new_layer, keeper)
    end
    # 不满足单边条件的节点也保留
    for n in layer
        isempty(n.in_edges) && isempty(n.out_edges) && continue
        direction == L2R() && length(n.in_edges) != 1 && push!(new_layer, n)
        direction == R2L() && length(n.out_edges) != 1 && push!(new_layer, n)
    end
    return new_layer
end