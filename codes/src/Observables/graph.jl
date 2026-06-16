ObservableGraph(L::Int64) = InteractionGraph(L,ObservableOperator)

function DirectedAcyclicGraph(tunnels::Vector{AbstractTunnel{L,T}}) where {L,T <: Union{ObservableOperator, CompositeObservableOperator}}
    isempty(tunnels) && return DirectedAcyclicGraph((), ())
    entry = sentinel(T)
    exit_s = sentinel(T)
    for tun in tunnels
        prev = entry
        for pos in 1:L
            val = tun[pos]
            node = DirectedNode(val)
            add_edge!(prev, node, ObservableWeight(pos == 1 ? tun.strength : 1.0))
            prev = node
        end
        add_edge!(prev, exit_s, ObservableWeight())
    end
    return DirectedAcyclicGraph((entry,), (exit_s,))
end

composite(A::InteractionGraph{L,ObservableOperator,DirectedAcyclicGraph{1,1}}, B::InteractionGraph{L,ObservableOperator,DirectedAcyclicGraph{1,1}}) where L = InteractionGraph([composite(a,b) for a in A.tunnel for b in B.tunnel])

# -- 初始化 InteractionGraph：建 DAG --
function initialize!(ig::InteractionGraph{L,T,DirectedAcyclicGraph{1,1}};verbose::Bool = false, N::Int64 = 1) where {L,T <: Union{ObservableOperator,CompositeObservableOperator}}
    to = TimerOutput()
    @assert !isempty(ig.tunnel) "No Interaction Tunnel!"
    @timeit to "build!" ig.graph = DirectedAcyclicGraph(ig.tunnel)
    @timeit to "optimize!" ig.graph,localto = optimize!(ig.graph;verbose = verbose, N = N)
    merge!(to,localto;tree_point = ["optimize!"])
    show(to;title = "Observable Graph"); print("\n"); show(ig); flush(stdout)
    ig.graph.source[1].val = T(0)
    ig.graph.sink[1].val = T(L + 1)
    ig.values = Dict{Tuple,Dict}()
    return ig
end

function setdefault!(ig::InteractionGraph{L,ObservableOperator,DirectedAcyclicGraph{1,1}}, obj::T) where {L, T <: Union{DenseMPS,DenseMPO}}
    # ig.graph.source[1].val.EnvL = LeftEnvironmentTensor(_left_isometry(obj))
    # ig.graph.sink[1].val.EnvR = RightEnvironmentTensor(_right_isometry(obj))
    EnvL = LeftEnvironmentTensor(_left_isometry(obj))
    EnvR = RightEnvironmentTensor(_right_isometry(obj))
    for e in ig.graph.source[1].out_edges
        e.weight.EnvL = EnvL
        e.weight.leftdata = Dict("site" => Int64[], "name" => String[])
    end
    for e in ig.graph.sink[1].in_edges
        e.weight.EnvR = EnvR
        e.weight.rightdata = Dict("site" => Int64[], "name" => String[])
    end
end

function setdefault!(ig::InteractionGraph{L,CompositeObservableOperator{N},DirectedAcyclicGraph{1,1}}, obj::T) where {L, T <: Union{DenseMPS,DenseMPO}, N}
    # ig.graph.source[1].val.EnvL = LeftEnvironmentTensor(_left_isometry(obj))
    # ig.graph.sink[1].val.EnvR = RightEnvironmentTensor(_right_isometry(obj))
    EnvL = LeftEnvironmentTensor(_left_isometry(obj))
    EnvR = RightEnvironmentTensor(_right_isometry(obj))
    for e in ig.graph.source[1].out_edges
        e.weight.EnvL = EnvL
        e.weight.leftdata = Dict("site" => [Int64[] for _ in 1:N], "name" => [String[] for _ in 1:N])
    end
    for e in ig.graph.sink[1].in_edges
        e.weight.EnvR = EnvR
        e.weight.rightdata = Dict("site" => [Int64[] for _ in 1:N], "name" => [String[] for _ in 1:N])
    end
end

function isdefaultweight(ig::InteractionGraph{L,T}) where {L, T <: Union{ObservableOperator,CompositeObservableOperator{2}}}
    for e in collect_edges(ig.graph)
        !isleftdefault(e) && return true
        !isrightdefault(e) && return true
    end
    return false
end
