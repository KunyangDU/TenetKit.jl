ObservableGraph(L::Int64) = InteractionGraph(L, LocalOperator, ObservableWeight)

composite(A::InteractionGraph{L,LocalOperator,ObservableWeight,DirectedAcyclicGraph{1,1}}, B::InteractionGraph{L,LocalOperator,ObservableWeight,DirectedAcyclicGraph{1,1}}) where L = InteractionGraph([composite(a,b) for a in A.tunnel for b in B.tunnel];W = ObservableWeight)

# -- 初始化 InteractionGraph：建 DAG --
function initialize!(ig::InteractionGraph{L,T,ObservableWeight,DirectedAcyclicGraph{1,1}};verbose::Bool = false, N::Int64 = 1) where {L,T <: Union{LocalOperator,CompositeLocalOperator}}
    to = TimerOutput()
    @assert !isempty(ig.tunnel) "No Interaction Tunnel!"
    @timeit to "build!" ig.graph,to′ = DirectedAcyclicGraph(ig.tunnel, ObservableWeight)
    merge!(to,to′;tree_point = ["build!"])
    show(to;title = "Observable Graph"); print("\n"); show(ig); flush(stdout)
    ig.graph.source[1].val = T(0)
    ig.graph.sink[1].val = T(L + 1)
    ig.values = Dict{Tuple,Dict}()
    return ig
end

function setdefault!(ig::InteractionGraph{L,LocalOperator,ObservableWeight,DirectedAcyclicGraph{1,1}}, obj::T) where {L, T <: Union{DenseMPS,DenseMPO}}
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

function setdefault!(ig::InteractionGraph{L,CompositeLocalOperator{N},ObservableWeight,DirectedAcyclicGraph{1,1}}, obj::T) where {L, T <: Union{DenseMPS,DenseMPO}, N}
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

function isdefaultweight(ig::InteractionGraph{L,T}) where {L, T <: Union{LocalOperator,CompositeLocalOperator{2}}}
    for e in collect_edges(ig.graph)
        !isleftdefault(e) && return false
        !isrightdefault(e) && return false
    end
    return true
end
