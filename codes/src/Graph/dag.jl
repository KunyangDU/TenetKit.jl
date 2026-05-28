mutable struct DirectedAcyclicGraph{E₁,E₂,T} <: AbstractGraph
    source::NTuple{E₁,T}
    sink::NTuple{E₂,T}
end

# ========================= DirectedAcyclicGraph 构造函数 =========================
# 从 tunnels 构建未优化的 DirectedAcyclicGraph
function DirectedAcyclicGraph(tunnels::Vector{InteractionTunnel{L,LocalOperator}}) where L
    isempty(tunnels) && return DirectedAcyclicGraph((), ())
    entry = sentinel(AbstractLocalOperator)
    exit_s = sentinel(AbstractLocalOperator)
    for tun in tunnels
        prev = entry
        for pos in 1:L
            val = tun[pos]
            node = DirectedNode(val)
            add_edge!(prev, node, pos == 1 ? tun.strength : 1.0)
            prev = node
        end
        add_edge!(prev, exit_s, 1.0)
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

# -- 枚举 source → sink 的全部路径 --
function paths(dag::DirectedAcyclicGraph)
    result = Vector{Vector{DirectedNode}}()
    _dfs_paths!(result, DirectedNode[], dag.source[1], dag.sink[1])
    return result
end

function _dfs_paths!(result, path, node, sink)
    push!(path, node)
    if node === sink
        push!(result, copy(path))
    else
        for e in node.out_edges
            _dfs_paths!(result, path, e.to, sink)
        end
    end
    pop!(path)
end

function Base.show(io::IO, dag::DirectedAcyclicGraph)
    layers = _extract_layers(dag)
    sizes = [length(l) for l in layers]
    print(io, "$(typeof(dag)): ", join(sizes, " → "))
end

graphsize(ig::DirectedAcyclicGraph) = length(collect_nodes(ig))
