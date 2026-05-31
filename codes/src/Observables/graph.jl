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
            add_edge!(prev, node, ObservableWeight())
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
    show(to;title = "Observable Graph"); print("\n"); flush(stdout)
    ig.graph.source[1].val = T(0)
    ig.graph.sink[1].val = T(L + 1)
    ig.values = Dict{Tuple,Dict}()
    return ig
end

function setdefault!(ig::InteractionGraph{L,ObservableOperator,DirectedAcyclicGraph{1,1}}, obj::T) where {L, T <: Union{DenseMPS,DenseMPO}}
    ig.graph.source[1].val.EnvL = LeftEnvironmentTensor(_left_isometry(obj))
    ig.graph.sink[1].val.EnvR = RightEnvironmentTensor(_right_isometry(obj))
    ig.graph.source[1].val.leftdata = Dict("site" => Int64[], "name" => String[])
    ig.graph.sink[1].val.rightdata = Dict("site" => Int64[], "name" => String[])
end

function setdefault!(ig::InteractionGraph{L,CompositeObservableOperator{N},DirectedAcyclicGraph{1,1}}, obj::T) where {L, T <: Union{DenseMPS,DenseMPO}, N}
    ig.graph.source[1].val.EnvL = LeftEnvironmentTensor(_left_isometry(obj))
    ig.graph.sink[1].val.EnvR = RightEnvironmentTensor(_right_isometry(obj))
    ig.graph.source[1].val.leftdata = Dict("site" => [Int64[] for _ in 1:N], "name" => [String[] for _ in 1:N])
    ig.graph.sink[1].val.rightdata = Dict("site" => [Int64[] for _ in 1:N], "name" => [String[] for _ in 1:N])
end
