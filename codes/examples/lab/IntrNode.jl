# 输入：A，包含多个长度相同的向量 (元素类型 T 可自定义)
# 输出：使得总节点数目最少的合并方式
# 合并规则：
#    左侧：从左端点开始逐点相同
#    右侧：从右端点开始逐点相同

# ========================= 数据结构 =========================
# 泛型 DAG 节点：双向边 + 哨兵标记 (val === nothing 表示哨兵)
mutable struct InteractionGraphNode{T}
    val::Union{T, Nothing}
    in::Vector{InteractionGraphNode}
    out::Vector{InteractionGraphNode}
end

# 构造器：普通节点
InteractionGraphNode(val::T) where T = InteractionGraphNode{T}(val, Vector{InteractionGraphNode}(), Vector{InteractionGraphNode}())
# 构造器：哨兵节点
sentinel(::Type{T}) where T = InteractionGraphNode{T}(nothing, Vector{InteractionGraphNode}(), Vector{InteractionGraphNode}())

issentinel(n::InteractionGraphNode) = n.val === nothing
isleaf(n::InteractionGraphNode)     = length(n.out) == 0
isfork(n::InteractionGraphNode)     = length(n.out) > 1
isjoin(n::InteractionGraphNode)     = length(n.in) > 1

# 添加有向边（防重复，不要求两端类型参数一致）
function add_edge!(from::InteractionGraphNode, to::InteractionGraphNode)
    to in from.out && return
    push!(from.out, to)
    push!(to.in, from)
end

# 值的显示表示（打印时使用，可重载）
_show_val(val) = sprint(show, val)

# ========================= 核心算法 =========================

# -- 序列 → 线性链 --
_chain(seq::Vector{T}) where T = _chain!(InteractionGraphNode.(seq))
function _chain(seq)
    n = length(seq)
    n == 0 && return (nothing, nothing)
    nodes = InteractionGraphNode[InteractionGraphNode(seq[i]) for i in 1:n]
    _chain!(nodes)
    return (nodes[1], nodes[end])
end

function _chain!(nodes::Vector{<:InteractionGraphNode})
    for i in 1:length(nodes)-1
        add_edge!(nodes[i], nodes[i+1])
    end
    return (nodes[1], nodes[end])
end

# -- LCP / LCS --
_find_lcp(seqs) = _find_lcp(seqs, length(seqs[1]))
function _find_lcp(seqs, n)
    for i in 1:n
        v = seqs[1][i]
        all(s -> isequal(s[i], v), seqs) || return i - 1
    end
    return n
end

function _find_lcs(seqs, n, lcp)
    for i in 1:(n - lcp)
        pos = n - i + 1
        v = seqs[1][pos]
        all(s -> isequal(s[pos], v), seqs) || return i - 1
    end
    return n - lcp
end

# -- 从参考序列抽取前缀/后缀链 --
function _prefix_chain(seqs, n, lcp)
    lcp == 0 && return InteractionGraphNode[]
    nodes = InteractionGraphNode[InteractionGraphNode(seqs[1][i]) for i in 1:lcp]
    _chain!(nodes)
    return nodes
end

function _suffix_chain(seqs, n, lcs)
    lcs == 0 && return InteractionGraphNode[]
    nodes = InteractionGraphNode[InteractionGraphNode(seqs[1][n-lcs+i]) for i in 1:lcs]
    _chain!(nodes)
    return nodes
end

# -- 中间段按首元素分组（不用 Dict，避免依赖 hash） --
function _group_by_first(mid_seqs::Vector{Vector{T}}) where T
    order  = T[]
    groups = Vector{Vector{T}}[]
    for s in mid_seqs
        if !isempty(s)
            k = s[1]
            idx = findfirst(x -> isequal(x, k), order)
            if idx === nothing
                push!(order, k)
                push!(groups, [s[2:end]])
            else
                push!(groups[idx], s[2:end])
            end
        end
    end
    return order, groups
end

# -- 构建各组子图 --
function _build_groups(order::Vector{T}, groups::Vector{Vector{Vector{T}}}) where T
    roots, tails = InteractionGraphNode[], InteractionGraphNode[]
    for (i, key) in enumerate(order)
        gs = groups[i]
        head = InteractionGraphNode(key)
        push!(roots, head)
        if isempty(gs[1])
            push!(tails, head)
        else
            sub_root, sub_tail = build_intrmap(gs)
            if sub_root !== nothing
                add_edge!(head, sub_root)
                push!(tails, sub_tail)
            end
        end
    end
    return roots, tails
end

# -- 确定根（单入口 / 哨兵多入口） --
function _root_or_sentinel(prefix::Vector{<:InteractionGraphNode}, group_roots::Vector{<:InteractionGraphNode})
    !isempty(prefix)       && return prefix[1]
    length(group_roots) == 1 && return group_roots[1]
    r = sentinel(typeof(group_roots[1].val))
    for gr in group_roots; add_edge!(r, gr); end
    return r
end

# -- 连接：前缀 → 分叉组；叶子 → 后缀 --
function _stitch!(prefix::Vector{<:InteractionGraphNode},
                  suffix::Vector{<:InteractionGraphNode},
                  group_roots::Vector{<:InteractionGraphNode},
                  group_tails::Vector{<:InteractionGraphNode})
    !isempty(prefix) && !isempty(group_roots) &&
        foreach(gr -> add_edge!(prefix[end], gr), group_roots)

    tails = !isempty(group_roots) ? group_tails : [prefix[end]]

    if !isempty(suffix)
        foreach(t -> add_edge!(t, suffix[1]), tails)
        return suffix[end]
    end

    length(tails) == 1 && return tails[1]
    # 多尾巴 → 出口哨兵，供从右向左遍历
    r = sentinel(typeof(tails[1].val))
    for t in tails
        add_edge!(t, r)
    end
    return r
end

"""
    build_intrmap(seqs::Vector{Vector{T}}) where T

递归构建合并 DAG，返回 `(left_root, right_root)`
两个入口均为单节点（可能是哨兵），支持从左或从右进入算法。
"""
function build_intrmap(seqs)
    isempty(seqs) && return (nothing, nothing)
    length(seqs) == 1 && return _chain(seqs[1])

    n = length(seqs[1])
    n == 0 && return (nothing, nothing)

    lcp = _find_lcp(seqs, n)
    lcs = _find_lcs(seqs, n, lcp)

    prefix   = _prefix_chain(seqs, n, lcp)
    suffix   = _suffix_chain(seqs, n, lcs)

    mid_seqs                 = [s[lcp+1 : n-lcs] for s in seqs]
    order, groups            = _group_by_first(mid_seqs)
    group_roots, group_tails = _build_groups(order, groups)

    isempty(prefix) && isempty(group_roots) && return (nothing, nothing)

    left_root  = _root_or_sentinel(prefix, group_roots)
    right_root = _stitch!(prefix, suffix, group_roots, group_tails)

    return (left_root, right_root)
end

# ========================= DAG 全局最小化 =========================
# 迭代精炼等价类：两个节点等价 ⇔ 值相同 ∧ 所有出边指向等价节点

function _collect_all!(node::InteractionGraphNode, result, seen)
    node in seen && return
    push!(seen, node)
    push!(result, node)
    for child in node.out
        _collect_all!(child, result, seen)
    end
end

function _eff_children(node::InteractionGraphNode)
    result = eltype(node.out)[]
    for c in node.out
        if issentinel(c)
            append!(result, _eff_children(c))
        else
            push!(result, c)
        end
    end
    return result
end

"""
    minimize!(root::InteractionGraphNode{T}) where T

全局最小化 DAG，合并等价节点，原地修改
"""
function minimize!(root::InteractionGraphNode)
    all_nodes = InteractionGraphNode[]
    _collect_all!(root, all_nodes, Set{InteractionGraphNode}())

    # 初始化等价类：按值分组（数组+findfirst，避免依赖 hash）
    val_order = Any[]
    val_ids   = Int[]
    class = Dict{InteractionGraphNode, Int}()
    for n in all_nodes
        issentinel(n) && continue
        idx = findfirst(x -> isequal(x, n.val), val_order)
        if idx === nothing
            push!(val_order, n.val)
            push!(val_ids, length(val_ids) + 1)
            class[n] = val_ids[end]
        else
            class[n] = val_ids[idx]
        end
    end

    # 迭代精炼至不动点
    while true
        sig2id = Dict{String, Int}()
        new_class = Dict{InteractionGraphNode, Int}()
        next_id = 1

        for n in all_nodes
            issentinel(n) && continue
            eff = _eff_children(n)
            child_ids = Int[class[c] for c in eff]
            sort!(child_ids)
            sig = _show_val(n.val) * "|" * join(child_ids, ",")

            id = get!(sig2id, sig) do
                id = next_id; next_id += 1; id
            end
            new_class[n] = id
        end

        stable = true
        for n in all_nodes
            issentinel(n) && continue
            if new_class[n] != get(class, n, 0)
                stable = false; break
            end
        end
        stable && break
        class = new_class
    end

    # 每类选代表
    rep = Dict{Int, InteractionGraphNode}()
    for n in all_nodes
        issentinel(n) && continue
        get!(rep, class[n], n)
    end

    # 重建边（两阶段：先算后清再连）
    new_edges = Dict{InteractionGraphNode, Vector{InteractionGraphNode}}()
    for n in all_nodes
        children = InteractionGraphNode[]
        for c in _eff_children(n)
            r = rep[class[c]]
            r in children || push!(children, r)
        end
        new_edges[n] = children
    end

    for n in all_nodes
        empty!(n.out); empty!(n.in)
    end
    for (n, children) in new_edges
        for c in children
            add_edge!(n, c)
        end
    end

    root
end

# ========================= 输出显示 =========================
# AbstractTrees 风格：├── / └── / │，线性链折叠，汇合 ─┘

function _collect_chain(node::InteractionGraphNode)
    vals = Any[node.val]
    cur = node
    while length(cur.out) == 1
        nxt = cur.out[1]
        issentinel(nxt) && break
        isjoin(nxt) && break
        cur = nxt
        push!(vals, cur.val)
    end
    return vals, cur
end

function _print_dag(node::InteractionGraphNode, prefix::String, is_last::Bool,
                    visited::Set)
    # 哨兵透明
    if issentinel(node)
        for (i, child) in enumerate(node.out)
            _print_dag(child, prefix, i == length(node.out), visited)
        end
        return
    end

    connector = is_last ? "└── " : "├── "

    # 汇合点已展开 → 引用
    if isjoin(node) && node in visited
        println(prefix, connector, _show_val(node.val), " ─┘")
        return
    end

    if isjoin(node)
        push!(visited, node)
    end

    # 收集并打印链
    chain_vals, tail = _collect_chain(node)
    chain_str = join(_show_val.(chain_vals), " ─ ")
    println(prefix, connector, chain_str)

    # 链前缀宽度补偿（子节点对齐到链末）
    chain_extra = 0
    if length(chain_vals) > 1
        pre_tail = join(_show_val.(chain_vals[1:end-1]), " ─ ") * " ─ "
        chain_extra = length(pre_tail)
    end

    n_out = length(tail.out)
    child_prefix = prefix * (is_last ? "    " : "│   ") * " "^chain_extra

    if n_out > 1
        for (i, child) in enumerate(tail.out)
            _print_dag(child, child_prefix, i == n_out, visited)
        end
    elseif n_out == 1
        child = tail.out[1]
        if issentinel(child)
            for (i, gc) in enumerate(child.out)
                _print_dag(gc, child_prefix, i == length(child.out), visited)
            end
        elseif isjoin(child) && child in visited
            println(child_prefix, "└── ", _show_val(child.val), " ─┘")
        else
            _print_dag(child, child_prefix, true, visited)
        end
    end
end

"""
    print_intrmap(root::InteractionGraphNode)

以树形图打印 IntrMap DAG
"""
function print_intrmap(root::InteractionGraphNode)
    _print_dag(root, "", true, Set{InteractionGraphNode}())
end

# ========================= 辅助函数 =========================
function count_nodes(node::InteractionGraphNode, visited=Set{InteractionGraphNode}())
    node in visited && return 0
    push!(visited, node)
    cnt = issentinel(node) ? 0 : 1
    for child in node.out
        cnt += count_nodes(child, visited)
    end
    return cnt
end

function count_edges(node::InteractionGraphNode, visited=Set{InteractionGraphNode}())
    node in visited && return 0
    push!(visited, node)
    cnt = length(node.out)
    for child in node.out
        cnt += count_edges(child, visited)
    end
    return cnt
end

# ========================= 运行示例 =========================
function demo(desc, seqs)
    println("=== ", desc, " ===")
    println("输入: ", seqs)
    r, _ = build_intrmap(seqs)
    n_greedy = count_nodes(r)
    minimize!(r)
    n_opt = count_nodes(r)
    print_intrmap(r)
    println("  原始: $(sum(length,s for s in seqs)) → 贪心: $n_greedy → 最小化: $n_opt")
    n_greedy > n_opt && println("  ⚡ 跨组合并了 $(n_greedy - n_opt) 个冗余节点")
    println()
end

demo("基本示例",
    [[1,1,1,1,2,1,1,3,1,1,1],
     [1,1,1,1,1,4,1,1,5,1,1]])

demo("跨组共享（反例）",
    [[1,2,3,4],
     [1,5,3,4],
     [6,2,3,4]])

demo("深度嵌套",
    [[1,2,3,4,5,6,7],
     [1,2,3,9,5,6,7],
     [1,2,8,4,5,6,7]])

# ========================= Tunnel 桥接 =========================
# 依赖 IntType.jl 先加载。将 InteractionTunnel 序列化后建图并赋值。
# 用法：ig = InteractionGraph(tunnels); build_graph!(ig)

if isdefined(@__MODULE__, :InteractionGraph)

"""
    build_graph!(ig::InteractionGraph{L}) where L

直接从 `ig.tunnel` 建图。Tunnel 实现了 `length` / `getindex`，
建图算法通过 duck typing 直接消费 Tunnel，无需预拷贝为 Vector。
"""
function build_graph!(ig::InteractionGraph{L}) where L
    left_root, right_root = build_intrmap(ig.tunnel)
    minimize!(left_root)
    ig.graph = SimpleGraph((left_root, right_root))
    return ig
end

end # @isdefined InteractionGraph
