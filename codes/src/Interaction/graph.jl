
mutable struct InteractionGraph{L,T,G <: DirectedAcyclicGraph{1,1}}
    graph::Union{Nothing, G}
    tunnel::Vector{InteractionTunnel{L,T}}
    values::Union{Nothing, Dict}
    L::Int64

    function InteractionGraph(tunnel::Vector{<:InteractionTunnel{L,T,N}}) where {L,T,N}
        return new{L, T, DirectedAcyclicGraph{1,1}}(nothing, tunnel, nothing, L)
    end

    function InteractionGraph(L::Int64, T::Type = LocalOperator)
        return new{L, T, DirectedAcyclicGraph{1,1}}(nothing, Vector{InteractionTunnel{L,T}}(), nothing, L)
    end
end

# -- 初始化 InteractionGraph：建 DAG --
function initialize!(ig::InteractionGraph{L, LocalOperator, DirectedAcyclicGraph{1,1}};verbose::Bool = false) where L
    # @assert !isempty(ig.tunnel) "No Interaction Tunnel!"
    # ig.graph = DirectedAcyclicGraph(ig.tunnel)
    # ig.graph = optimize!(ig.graph)
    to = TimerOutput()
    @assert !isempty(ig.tunnel) "No Interaction Tunnel!"
    @timeit to "build!" ig.graph = DirectedAcyclicGraph(ig.tunnel)
    @timeit to "optimize!" ig.graph,localto = optimize!(ig.graph)
    merge!(to,localto;tree_point = ["optimize!"])
    verbose && (show(to;title = "Interaction Graph"); print("\n"); flush(stdout))
    return ig
end


function Base.show(io::IO, ig::InteractionGraph{L}) where L
    print(io, "$(typeof(ig)) $(100*(1 - round((isnothing(ig.graph) ? length(ig.tunnel) * L : graphsize(ig.graph)) / length(ig.tunnel) / L;digits = 4)))% compressed\n")
    println(io, " - ","graph : ", isnothing(ig.graph) ? Nothing : ig.graph)
    println(io, " - ","tunnel: $(length(ig.tunnel)) x $(typeof(ig.tunnel))")
    println(io, " - ","values: $(typeof(ig.values)) -> $(isnothing(ig.values) ? 0 : dictsize(ig.values)) elements")
end


