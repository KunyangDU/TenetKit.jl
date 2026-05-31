# ========================= 图操作 =========================


# -- weight 继承：仅当 bond weight 冲突时才推系数 --
# 合并时 other 的共享边会被删除，若 keeper 同位置 weight 相同则直接覆盖，无需推；
# 仅当 weight 不同 (冲突) 时才将各自 weight 推到非共享边，共享边统一重置为 1。
function _merge_inherit_weight!(keeper::DirectedNode, other::DirectedNode, ::L2R)
    # 两边入边 weight 都是 default → 无冲突，跳过
    all_default = all(isdefault, keeper.in_edges) && all(isdefault, other.in_edges)
    all_default && return
    for n in (keeper, other)
        @assert length(n.in_edges) == 1
        for in_e in n.in_edges
            w = in_e.weight
            isdefault(in_e) && continue
            for out_e in n.out_edges
                inherit_weight!(out_e, w)
            end
            default_weight!(in_e)
        end
    end
end

function _merge_inherit_weight!(keeper::DirectedNode, other::DirectedNode, ::R2L)
    all_default = all(isdefault, keeper.out_edges) && all(isdefault, other.out_edges)
    all_default && return
    for n in (keeper, other)
        @assert length(n.out_edges) == 1
        for out_e in n.out_edges
            w = out_e.weight
            isdefault(out_e) && continue
            for in_e in n.in_edges
                inherit_weight!(in_e, w)
            end
            default_weight!(out_e)
        end
    end
end

# 外部锁表：按 node objectid 按需分配 ReentrantLock
const _node_locks = Dict{UInt, ReentrantLock}()
const _locks_guard = ReentrantLock()

function _get_node_lock(node::DirectedNode)
    oid = objectid(node)
    lock(_locks_guard) do
        get!(() -> ReentrantLock(), _node_locks, oid)
    end
end

function _merge_into!(keeper::DirectedNode, other::DirectedNode, direction = L2R())
    to = TimerOutput()
    @timeit to "_merge_inherit_weight!" _merge_inherit_weight!(keeper, other, direction)
    @timeit to "edit edge" if direction == L2R()
        for e in copy(other.out_edges)
            lock(_get_node_lock(e.to)) do
                _remove_edge!(other, e.to)
                add_edge!(keeper, e.to, e.weight)
            end
        end
        for e in copy(other.in_edges)
            lock(_get_node_lock(e.from)) do
                _remove_edge!(e.from, other)
            end
        end
    else
        for e in copy(other.out_edges)
            lock(_get_node_lock(e.to)) do
                _remove_edge!(other, e.to)
            end
        end
        for e in copy(other.in_edges)
            lock(_get_node_lock(e.from)) do
                _remove_edge!(e.from, other)
                add_edge!(e.from, keeper, e.weight)
            end
        end
    end
    empty!(other.out_edges)
    empty!(other.in_edges)
    return to
end

function _merge_by!(sig_fn::Function, layer::Vector{<:DirectedNode}, direction = L2R())
    to = TimerOutput()
    groups = Dict{Any, Vector{DirectedNode}}()
    new_layer = DirectedNode[]
    @timeit to "find group" for n in layer
        isempty(n.in_edges) && isempty(n.out_edges) && continue
        mergeable = direction == L2R() ? (length(n.in_edges) == 1) : (length(n.out_edges) == 1)
        if mergeable
            sig = sig_fn(n)
            vec = get!(groups, sig) do; DirectedNode[]; end
            push!(vec, n)
        else
            push!(new_layer, n)
        end
    end
    glist = collect(values(groups))
    isempty(glist) && return new_layer, to
    results = Vector{DirectedNode}(undef, length(glist))
    cnt  = Threads.Atomic{Int64}(1)
    @timeit to "_merge_into!" Threads.@threads for _ in 1:get_nworker()
        while true
            i = Threads.atomic_add!(cnt, 1)
            i > length(glist) && break
            group = glist[i]
            keeper = group[1]
            for n in group[2:end]
                _merge_into!(keeper, n, direction)
            end
            results[i] = keeper
        end
    end
    append!(new_layer, results)
    return new_layer, to
end