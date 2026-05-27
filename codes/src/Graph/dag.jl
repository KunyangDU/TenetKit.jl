mutable struct DirectedAcyclicGraph{E₁,E₂,T} <: AbstractGraph
    source::NTuple{E₁,T}
    sink::NTuple{E₂,T}
end

# ========================= DirectedAcyclicGraph 构造函数 =========================
# 从 tunnels 构建未优化的 DirectedAcyclicGraph
function DirectedAcyclicGraph(tunnels::Vector)
    isempty(tunnels) && return DirectedAcyclicGraph((), ())
    L = length(tunnels[1])
    entry = sentinel(AbstractLocalOperator)
    exit_s = sentinel(AbstractLocalOperator)
    for tun in tunnels
        prev = entry
        for pos in 1:L
            val = tun[pos]
            node = DirectedNode(val)
            w = pos == 1 ? tun.strength : 1.0
            add_edge!(prev, node; weight = w)
            prev = node
        end
        add_edge!(prev, exit_s)
    end
    return DirectedAcyclicGraph((entry,), (exit_s,))
end

# -- 从 DAG source 出发收集全部节点 --
function collect_nodes(dag::DirectedAcyclicGraph)
    all_nodes = Vector{DirectedNode}()
    seen = Set{DirectedNode}()
    queue = collect(Any, dag.source)
    while !isempty(queue)
        n = popfirst!(queue)
        n in seen && continue
        push!(all_nodes, n)
        push!(seen, n)
        for e in n.out_edges
            e.to in seen || push!(queue, e.to)
        end
    end
    return all_nodes
end
