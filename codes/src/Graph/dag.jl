mutable struct DirectedAcyclicGraph{E₁,E₂,T} <: AbstractGraph
    source::NTuple{E₁,T}
    sink::NTuple{E₂,T}
end

# 从 tunnels 构建未优化的 DirectedAcyclicGraph
function DirectedAcyclicGraph(tunnels::Vector{AbstractTunnel{L,T}}; weight::Type{W} = Number) where {L,T,W}
    isempty(tunnels) && return DirectedAcyclicGraph((), ())
    entry = sentinel(AbstractLocalOperator)
    exit_s = sentinel(AbstractLocalOperator)
    for tun in tunnels
        prev = entry
        for pos in 1:L
            val = tun[pos]
            node = DirectedNode(val)
            add_edge!(prev, node, W(pos == 1 ? tun.strength : 1.0))
            prev = node
        end
        add_edge!(prev, exit_s, W(1.0))
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

# -- 收集全部边 --
function collect_edges(dag::DirectedAcyclicGraph)
    edges = DirectedEdge[]
    seen = Set{DirectedNode}()
    queue = collect(Any, dag.source)
    while !isempty(queue)
        n = popfirst!(queue)
        n in seen && continue
        push!(seen, n)
        for e in n.out_edges
            push!(edges, e)
            e.to in seen || push!(queue, e.to)
        end
    end
    return edges
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


function composite(A::DirectedAcyclicGraph, B::DirectedAcyclicGraph)
    # cause errors for calObs by introducing multi in/out edges
    layersA = _extract_layers(A)
    layersB = _extract_layers(B)
    @assert (L = length(layersA)) == length(layersB) "DAGs must have same number of layers, got $(length(layersA)) vs $(length(layersB))"
    
    layers = Vector{DirectedNode}[]
    mapping = Dict{NTuple{2,DirectedNode},DirectedNode}()
    for i in 1:L
        layer = DirectedNode[]
        lA = layersA[i]
        lB = layersB[i]
        tmp = Dict{NTuple{2,DirectedNode},DirectedNode}()
        for nA in lA, nB in lB
            !haskey(mapping,(nA,nB)) && (mapping[(nA,nB)] = DirectedNode(composite(nA.val,nB.val)))
            for (eA,cA) in child(nA), (eB,cB) in child(nB)
                !haskey(tmp,(cA,cB)) && (tmp[(cA,cB)] = DirectedNode(composite(cA.val,cB.val)))
                add_edge!(mapping[(nA,nB)],tmp[(cA,cB)],composite(eA.weight,eB.weight))
            end
            push!(layer,mapping[(nA,nB)])
        end
        push!(layers,layer)
        mapping = tmp
    end
    return DirectedAcyclicGraph(Tuple(layers[1]),Tuple(layers[end]))
end