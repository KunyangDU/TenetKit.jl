# ============================================================
# Myhill-Nerode 最优 DAG 构造
# 从 tunnels 直接生成最优 DAG（替换旧版先建链再 optimize）
# ============================================================

# -- 单个预分组的处理（纯函数，无副作用） --
function _process_group!(tunnels::Vector{<: AbstractTunnel}, tis::Vector{Int64},
                         l::Int, L::Int)
    to = TimerOutput()
    m::Int = length(tis)
    T′ = first(tunnels) isa InteractionTunnel ? InteractionTunnelSegment : CompositeInteractionTunnelSegment

    @timeit to "labels" begin
    local labels = Vector{Tuple{T′, T′}}(undef, m)
    for (i, ti) in enumerate(tis)
        tun = tunnels[ti]
        labels[i] = (T′(tun, 1, l-1),
                     T′(tun, l+1, L))
    end
    end  # labels

    @timeit to "bipartite" begin
        gc = BipartiteGraph(labels)
        @timeit to "initialize!" initialize!(gc, labels)
        @timeit to "max_matching!" max_matching!(gc)
        @timeit to "min_vertex_cover!" min_vertex_cover!(gc)
    end  # bipartite

    @timeit to "cover_extract" begin
    L2idx = Dict{T′, Vector{Int}}()
    R2idx = Dict{T′, Vector{Int}}()
    for (i, (ll, rr)) in enumerate(labels)
        push!(get!(L2idx, ll, Int[]), i)
        push!(get!(R2idx, rr, Int[]), i)
    end

    groups = Vector{Vector{Int64}}()
    assigned = falses(m)
    for iu in 1:length(gc.u)
        gc.coverL[iu] || continue
        node_tis = Int64[]
        for i in get(L2idx, gc.u[iu], Int[])
            assigned[i] && continue
            push!(node_tis, tis[i])
            assigned[i] = true
        end
        isempty(node_tis) || push!(groups, node_tis)
    end
    for jv in 1:length(gc.v)
        gc.coverR[jv] || continue
        node_tis = Int64[]
        for i in get(R2idx, gc.v[jv], Int[])
            assigned[i] && continue
            push!(node_tis, tis[i])
            assigned[i] = true
        end
        isempty(node_tis) || push!(groups, node_tis)
    end
    end  # cover_extract
    return groups, to
end

# -- 串行版本 --
function _group_serial!(tunnels::Vector{<:AbstractTunnel}, algo::Myhillalgo, to::TimerOutput)
    L = length(tunnels[1])
    K = length(tunnels)
    algo.nodes = Vector{Vector{Int64}}[Vector{Int64}[] for _ in 1:L]

    for l in 1:L
        @timeit to "pre_group" begin
        pre_groups = Dict{AbstractLocalOperator, Vector{Int64}}()
        for ti in 1:K
            key = tunnels[ti][l]
            push!(get!(pre_groups, key, Int64[]), ti)
        end
        end  # pre_group

        @timeit to "classify" begin
        for (_, tis) in pre_groups
            groups, to_pg = _process_group!(tunnels, tis, l, L)
            merge!(to, to_pg;tree_point = ["classify"])
            for g in groups
                push!(algo.nodes[l], g)
            end
        end
        end  # classify
    end
    return algo
end

# -- 并行版本 --
function _group_parallel!(tunnels::Vector{<:AbstractTunnel}, algo::Myhillalgo, to::TimerOutput)
    L = length(tunnels[1])
    K = length(tunnels)
    algo.nodes = Vector{Vector{Int64}}[Vector{Int64}[] for _ in 1:L]

    # 收集所有工作量：(layer, tunnel_indices)
    tasks = Tuple{Int,Vector{Int64}}[]
    for l in 1:L
        @timeit to "pre_group" begin
        pre_groups = Dict{AbstractLocalOperator, Vector{Int64}}()
        for ti in 1:K
            key = tunnels[ti][l]
            push!(get!(pre_groups, key, Int64[]), ti)
        end
        for (_, tis) in pre_groups
            push!(tasks, (l, tis))
        end
        end  # pre_group
    end

    nworker = get_num_threads_julia()
    N = length(tasks)
    lk   = Threads.ReentrantLock()
    cnt  = Threads.Atomic{Int64}(1)

    @timeit to "classify" begin
    Threads.@sync for _ in 1:nworker
        Threads.@spawn while true
            i = Threads.atomic_add!(cnt, 1)
            i > N && break
            l, tis = tasks[i]
            groups, to_pg = _process_group!(tunnels, tis, l, L)
            lock(lk)
            try
                merge!(to, to_pg;tree_point = ["classify"])
                for g in groups
                    push!(algo.nodes[l], g)
                end
            finally
                unlock(lk)
            end
        end
    end
    end  # classify

    return algo
end

# -- 入口：按线程数分发 --
function group!(tunnels::Vector{<:AbstractTunnel}, algo::Myhillalgo)
    to = TimerOutput()
    if get_num_threads_julia() ≤ 1
        _group_serial!(tunnels, algo, to)
    else
        _group_parallel!(tunnels, algo, to)
    end
    return algo, to
end

# -- 建图：nodes + tunnels → DirectedAcyclicGraph --
function build_dag(tunnels::Vector{<:AbstractTunnel}, algo::Myhillalgo)
    to = TimerOutput()
    L = length(tunnels[1])
    K = length(tunnels)
    W = algo.weight

    entry  = sentinel(AbstractLocalOperator)
    exit_s = sentinel(AbstractLocalOperator)

    @timeit to "nodes" begin
    tun2node = [Dict{Int,Int}() for _ in 1:L]
    dag_nodes = [DirectedNode[] for _ in 1:L]
    for l in 1:L
        for (nid, tis) in enumerate(algo.nodes[l])
            push!(dag_nodes[l], DirectedNode(tunnels[tis[1]][l]))
            for ti in tis
                tun2node[l][ti] = nid
            end
        end
    end
    end  # nodes

    for n in dag_nodes[1]; add_edge!(entry, n, W(1.0)); end

    @timeit to "edges" begin
    used = falses(K)
    for l in 1:L-1
        edge_tuns = Dict{Tuple{Int,Int}, Vector{Int}}()
        for ti in 1:K
            a = get(tun2node[l], ti, 0)
            b = get(tun2node[l+1], ti, 0)
            a == 0 || b == 0 && continue
            push!(get!(edge_tuns, (a, b), Int[]), ti)
        end
        for ((a, b), tis) in edge_tuns
            w_val = 1.0
            if length(tis) == 1
                ti = tis[1]
                if !used[ti]
                    w_val = tunnels[ti].strength
                    used[ti] = true
                end
            end
            add_edge!(dag_nodes[l][a], dag_nodes[l+1][b], W(w_val))
        end
    end
    for n in dag_nodes[L]; add_edge!(n, exit_s, W(1.0)); end
    end  # edges
    return DirectedAcyclicGraph((entry,), (exit_s,)), to
end

# -- 主入口：从 tunnels 直接生成最优 DAG --
function DirectedAcyclicGraph(tunnels::Vector{AbstractTunnel{L,T}}, algo::Myhillalgo) where {L,T}
    to = TimerOutput()
    @timeit to "group!" _, to1 = group!(tunnels, algo)
    merge!(to, to1; tree_point = ["group!"])
    @timeit to "build_dag" dag, to2 = build_dag(tunnels, algo)
    merge!(to, to2; tree_point = ["build_dag"])
    return dag, to
end
