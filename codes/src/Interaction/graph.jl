
mutable struct InteractionGraph{L,G<:AbstractGraph}
    graph::Union{Nothing, G}
    tunnel::Vector{<:InteractionTunnel{L}}
    values::Union{Nothing, Dict}
    L::Int64

    function InteractionGraph(tunnel::Vector{<:InteractionTunnel{L}}) where L
        return new{L, LayerDirectedAcyclicGraph}(nothing, tunnel, nothing, L)
    end

    function InteractionGraph(L::Int64)
        return new{L, LayerDirectedAcyclicGraph}(nothing, Vector{InteractionTunnel{L}}(), nothing, L)
    end
end

# -- 图构建：从 tunnels 建立并优化 DAG --
function build_graph!(ig::InteractionGraph{L}) where L
    isempty(ig.tunnel) && return ig
    ig.graph = optimize!(LayerDirectedAcyclicGraph(ig.tunnel))
    return ig
end

# -- 初始化 InteractionGraph：建 DAG --
function initialize!(ig::InteractionGraph)
    isnothing(ig.graph) && build_graph!(ig)
    return ig
end

