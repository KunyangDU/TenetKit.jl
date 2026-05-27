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
            add_edge!(prev, node)
            prev = node
        end
        add_edge!(prev, exit_s)
    end
    return DirectedAcyclicGraph((entry,), (exit_s,))
end
