# 二分图最小点覆盖 = 最大匹配 (König 定理)
# 左顶点 = 不同的左标签值，右顶点 = 不同的右标签值，每个元素 = 一条边
# 选一个顶点 = 以该标签建一个组

# ============================================================
# 分类器结构
# ============================================================
mutable struct BipartiteGraph{T} <: AbstractGraph
    u::Vector{T}
    v::Vector{T}
    adj::Vector{Vector{Int}}
    matchL::Vector{Int}
    matchR::Vector{Int}
    coverL::Vector{Bool}
    coverR::Vector{Bool}
    BipartiteGraph(A::Vector{<:Tuple}) = new{eltype(first(A))}([], [], [], [], [], [], [])
end

# ============================================================
# initialize! : 映射标签值 + 建邻接表
# ============================================================
function initialize!(gc::BipartiteGraph{T}, A::Vector{<:Tuple}) where T
    L = Dict{T,Int}()
    R = Dict{T,Int}()
    u_list = T[]
    v_list = T[]
    adj_buf = Int[]

    for (l, r) in A
        li = get!(L, l) do; push!(u_list, l); length(u_list); end
        ri = get!(R, r) do; push!(v_list, r); length(v_list); end
        push!(adj_buf, li, ri)
    end

    gc.u = u_list; gc.v = v_list
    nl, nr = length(u_list), length(v_list)

    gc.adj = [Int[] for _ in 1:nl]
    for k in 1:2:length(adj_buf)
        push!(gc.adj[adj_buf[k]], adj_buf[k+1])
    end

    gc.matchL = zeros(Int, nl)
    gc.matchR = zeros(Int, nr)
    gc.coverL = fill(false, nl)
    gc.coverR = fill(false, nr)
end

# ============================================================
# max_matching! : 匈牙利算法求最大匹配
# ============================================================
function max_matching!(gc::BipartiteGraph)
    nr = length(gc.v)
    gc.matchL .= 0
    gc.matchR .= 0

    function dfs(u, seen)
        for v in gc.adj[u]
            seen[v] && continue
            seen[v] = true
            if gc.matchR[v] == 0 || dfs(gc.matchR[v], seen)
                gc.matchR[v] = u
                gc.matchL[u] = v
                return true
            end
        end
        return false
    end

    matching = 0
    for u in 1:length(gc.u)
        if dfs(u, falses(nr))
            matching += 1
        end
    end
    return matching
end

# ============================================================
# min_vertex_cover! : König 构造最小点覆盖
# ============================================================
function min_vertex_cover!(gc::BipartiteGraph)
    nl, nr = length(gc.u), length(gc.v)
    reachableL = falses(nl)
    reachableR = falses(nr)
    stack = Int[]

    for u in 1:nl
        if gc.matchL[u] == 0
            reachableL[u] = true
            push!(stack, u)
        end
    end

    while !isempty(stack)
        u = pop!(stack)
        for v in gc.adj[u]
            reachableR[v] && continue
            reachableR[v] = true
            w = gc.matchR[v]
            if w != 0 && !reachableL[w]
                reachableL[w] = true
                push!(stack, w)
            end
        end
    end

    for u in 1:nl
        gc.coverL[u] = !reachableL[u]
    end
    for v in 1:nr
        gc.coverR[v] = reachableR[v]
    end
    return gc
end

# ============================================================
# build_groups : 从点覆盖结果构建分组
# ============================================================
function build_groups(gc::BipartiteGraph, A::Vector{<:Tuple})
    chosen = Tuple{Symbol,eltype(gc.u)}[]
    for i in 1:length(gc.u)
        gc.coverL[i] && push!(chosen, (:L, gc.u[i]))
    end
    for j in 1:length(gc.v)
        gc.coverR[j] && push!(chosen, (:R, gc.v[j]))
    end

    assigned = falses(length(A))
    T = eltype(A)
    groups = Vector{Tuple{Symbol,eltype(gc.u),Vector{T}}}()

    for (side, val) in chosen
        members = T[]
        for (i, (l, r)) in enumerate(A)
            if assigned[i]; continue; end
            if (side == :L && isequal(l, val)) || (side == :R && isequal(r, val))
                push!(members, (l, r))
                assigned[i] = true
            end
        end
        push!(groups, (side, val, members))
    end
    return groups
end

# ============================================================
# 便捷接口：一步完成分类
# ============================================================
function classify(A::Vector{<:Tuple})
    gc = BipartiteGraph(A)
    initialize!(gc, A)
    max_matching!(gc)
    min_vertex_cover!(gc)
    groups = build_groups(gc, A)
    return gc, groups
end
