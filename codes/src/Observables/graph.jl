ObservableGraph(L::Int64) = InteractionGraph(L,ObservableOperator)

# -- 初始化 InteractionGraph：建 DAG --
function initialize!(ig::InteractionGraph{L,ObservableOperator,DirectedAcyclicGraph{1,1}};verbose::Bool = true) where L
    to = TimerOutput()
    @assert !isempty(ig.tunnel) "No Interaction Tunnel!"
    @timeit to "build!" ig.graph = DirectedAcyclicGraph(ig.tunnel)
    @timeit to "optimize!" ig.graph,localto = optimize!(ig.graph)
    merge!(to,localto;tree_point = ["optimize!"])
    verbose && (show(to;title = "Observable Graph"); print("\n"); flush(stdout))
    ig.graph.source[1].val = ObservableOperator(0)
    ig.graph.sink[1].val = ObservableOperator(L + 1)
    ig.values = Dict{Tuple,Dict}()
    return ig
end

function setdefault!(ig::InteractionGraph{L,ObservableOperator,DirectedAcyclicGraph{1,1}}, obj::T) where {L, T <: Union{DenseMPS,DenseMPO}}
    ig.graph.source[1].val.EnvL = LeftEnvironmentTensor(_left_isometry(obj))
    ig.graph.sink[1].val.EnvR = RightEnvironmentTensor(_right_isometry(obj))
    ig.graph.source[1].val.leftdata = Dict("site" => Int64[], "name" => String[])
    ig.graph.sink[1].val.rightdata = Dict("site" => Int64[], "name" => String[])
end

function DirectedAcyclicGraph(tunnels::Vector{InteractionTunnel{L,ObservableOperator}}) where L
    isempty(tunnels) && return DirectedAcyclicGraph((), ())
    entry = sentinel(ObservableOperator{0,0})
    exit_s = sentinel(ObservableOperator{0,0})
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

