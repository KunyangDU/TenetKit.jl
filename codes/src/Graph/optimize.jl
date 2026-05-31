
# ========================= 逐层优化 =========================
_lr_sig(n::DirectedNode) = (Tuple(sort!([objectid(e.from) for e in n.in_edges])), string(n.val))
_rl_sig(n::DirectedNode) = (Tuple(sort!([objectid(e.to) for e in n.out_edges])), string(n.val))

# 单层 merge
function _lr_merge_at!(layers::Vector{Vector{DirectedNode}}, pos::Int)
    layers[pos],to = _merge_by!(_lr_sig, layers[pos], L2R())
    return to 
end
function _rl_merge_at!(layers::Vector{Vector{DirectedNode}}, pos::Int)
    layers[pos],to = _merge_by!(_rl_sig, layers[pos], R2L())
    return to 
end

# L→R 单层 split：多 in 节点每个 parent 拆到新建节点
function _lr_split_at!(layers::Vector{Vector{DirectedNode}}, pos::Int)
    layer = layers[pos]
    changed = false
    for n in layer
        length(n.in_edges) ≤ 1 && continue
        issentinel(n) && continue
        isempty(n.out_edges) && continue
        n in layer || continue
        for parent_edge in copy(n.in_edges)
            parent = parent_edge.from
            issentinel(parent) && continue
            length(n.in_edges) <= 1 && break
            m = DirectedNode(copy(n.val))
            push!(layer, m)
            _remove_edge!(parent, n)
            add_edge!(parent, m, deepcopy(parent_edge.weight))
            for out_e in n.out_edges
                add_edge!(m, out_e.to, deepcopy(out_e.weight))
            end
            changed = true
        end
    end
    return changed
end

# R→L 单层 split：多 out 节点每个 child 拆到新建节点
function _rl_split_at!(layers::Vector{Vector{DirectedNode}}, pos::Int)
    layer = layers[pos]
    changed = false
    for n in layer
        length(n.out_edges) ≤ 1 && continue
        issentinel(n) && continue
        isempty(n.in_edges) && continue
        n in layer || continue
        for child_edge in copy(n.out_edges)
            child = child_edge.to
            issentinel(child) && continue
            length(n.out_edges) <= 1 && break
            m = DirectedNode(copy(n.val))
            push!(layer, m)
            _remove_edge!(n, child)
            add_edge!(m, child, deepcopy(child_edge.weight))
            for in_e in n.in_edges
                add_edge!(in_e.from, m, deepcopy(in_e.weight))
            end
            changed = true
        end
    end
    return changed
end

# 逐层优化：split 拆开 (保证 merge 时 push 安全) → merge 推 weight → 反向 merge 合回去
function _l2r!(layers::Vector{Vector{DirectedNode}}, isrev::Bool;verbose::Bool = false)
    to = TimerOutput()
    for pos in 1:length(layers)
        sweepto = TimerOutput()
        @timeit sweepto "split!" _lr_split_at!(layers, pos)
        @timeit sweepto "merge!" localto = _lr_merge_at!(layers, pos)
        @timeit sweepto "rev_merge!" isrev && _rl_merge_at!(layers, pos)   # 合回被 in_set 拆开的节点
        merge!(sweepto,localto;tree_point = ["merge!"])
        merge!(to,sweepto)
        verbose && (show(sweepto,title = "_l2r! - $(pos) / $(length(layers))");print("\n"))
    end
    return to
end

function _r2l!(layers::Vector{Vector{DirectedNode}}, isrev::Bool;verbose::Bool = false)
    to = TimerOutput()
    for pos in reverse(1:length(layers))
        sweepto = TimerOutput()
        @timeit sweepto "split!" _rl_split_at!(layers, pos)
        @timeit sweepto "merge!" localto = _rl_merge_at!(layers, pos)
        @timeit sweepto "rev_merge!" isrev && _lr_merge_at!(layers, pos)   # 合回被 out_set 拆开的节点
        merge!(sweepto,localto;tree_point = ["merge!"])
        merge!(to,sweepto)
        verbose && (show(sweepto,title = "_r2l! - $(length(layers) - pos + 1) / $(length(layers))");print("\n"))
    end
    return to
end

function _lr2c!(layers::Vector{Vector{DirectedNode}}, isrev::Bool;verbose::Bool = false)
    to = TimerOutput()
    L′ = div(length(layers),2)
    for pos in 1:L′
        sweepto = TimerOutput()
        @timeit sweepto "split!" _lr_split_at!(layers, pos)
        @timeit sweepto "merge!" _lr_merge_at!(layers, pos)
        @timeit sweepto "rev_merge!" isrev && _rl_merge_at!(layers, pos)   # 合回被 in_set 拆开的节点
        merge!(to,sweepto)
        verbose && (show(sweepto,title = "_l2c! - $(pos) / $(L′)");print("\n"))
    end
    for pos in reverse(L′+1:length(layers))
        sweepto = TimerOutput()
        @timeit sweepto "split!" _rl_split_at!(layers, pos)
        @timeit sweepto "merge!" _rl_merge_at!(layers, pos)
        @timeit sweepto "rev_merge!" isrev && _lr_merge_at!(layers, pos)   # 合回被 out_set 拆开的节点
        merge!(to,sweepto)
        verbose && (show(sweepto,title = "_r2c! - $(length(layers) - pos + 1) / $(L′)");print("\n"))
    end
    return to
end

# ========================= 层提取 =========================
# 从 DAG 的 source 出发，逐层收集物理节点
function _extract_layers(dag::DirectedAcyclicGraph)
    layers = Vector{DirectedNode}[]
    cur = [dag.source[1]]
    while true
        push!(layers, cur)
        nxt = _next_nodes(cur)
        isempty(nxt) && break
        cur = nxt
    end
    return layers
end

function clear!(layers::Vector{Vector{DirectedNode}})
    for layer in layers
        filter!(n -> !(isempty(n.in_edges) && isempty(n.out_edges)), layer)
    end
end

# ========================= 优化入口 =========================
"""
    optimize!(dag::DirectedAcyclicGraph)

对 DirectedAcyclicGraph 运行 Moore 框架下的逐层 merge-split 优化，迭代至不动点。
"""
function optimize!(dag::DirectedAcyclicGraph; N::Int64 = 1, verbose::Bool = false)
    BLAS_MIN!()

    to = TimerOutput()
    @timeit to "collect layer" layers = _extract_layers(dag)
    isempty(layers) && return dag, to

    nlayers = length(layers)
    sizes   = Vector{Int}(undef, nlayers)
    for i in 1:N
        for (j, l) in enumerate(layers); sizes[j] = length(l); end
        @timeit to "sweep!" begin
            @timeit to "_l2r!"  to_l2r = _l2r!(layers, i ≠ 1;verbose = verbose)
            @timeit to "_r2l!"  to_r2l = _r2l!(layers, true;verbose = verbose)
            i == 1 && @timeit to "clear!" clear!(layers)
            @timeit to "_lr2c!" to_lr2c = _lr2c!(layers, true;verbose = verbose)
        end
        merge!(to,to_l2r;tree_point = ["sweep!","_l2r!"])
        merge!(to,to_r2l;tree_point = ["sweep!","_r2l!"])
        merge!(to,to_lr2c;tree_point = ["sweep!","_lr2c!"])
        converged = true
        for (j, l) in enumerate(layers)
            length(l) == sizes[j] || (converged = false; break)
        end
        converged && break
    end

    BLAS_MAX!()
    return dag, to
end
