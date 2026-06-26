
mutable struct InteractionGraph{L, T, W, G <: DirectedAcyclicGraph{1,1}}
    graph::Union{Nothing, G}
    tunnel::Vector{AbstractTunnel{L,T}}
    values::Union{Nothing, Dict}
    L::Int64

    function InteractionGraph(tunnel::Vector{<:AbstractTunnel{L,T}}; W::Type = Number) where {L,T}
        return new{L, T, W, DirectedAcyclicGraph{1,1}}(nothing, tunnel, nothing, L)
    end

    function InteractionGraph(L::Int64, T::Type = LocalOperator, W::Type = Number)
        return new{L, T, W, DirectedAcyclicGraph{1,1}}(nothing, Vector{AbstractTunnel{L,T}}(), nothing, L)
    end
end

# -- 初始化 InteractionGraph：建 DAG --
function initialize!(ig::InteractionGraph{L, T, Number, DirectedAcyclicGraph{1,1}};verbose::Bool = false, N::Int64 = 1) where {L, T <: Union{LocalOperator, CompositeLocalOperator}}
    to = TimerOutput()
    @assert !isempty(ig.tunnel) "No Interaction Tunnel!"
    @timeit to "build!" ig.graph,to′ = DirectedAcyclicGraph(ig.tunnel,Number)
    merge!(to,to′;tree_point = ["build!"])
    verbose && (show(to;title = "Interaction Graph"); print("\n"); show(ig); flush(stdout))
    return ig
end


function Base.show(io::IO, ig::InteractionGraph{L}) where {L}
    GS = isnothing(ig.graph) ? length(ig.tunnel) * L : graphsize(ig.graph)-2
    print(io, "$(typeof(ig)) (1 - $(GS)/$(length(ig.tunnel) * L)) ≈ $(round(100*(1 - GS / length(ig.tunnel) / L);digits = 4))% compressed\n")
    println(io, " - ","graph : ", isnothing(ig.graph) ? Nothing : ig.graph)
    println(io, " - ","tunnel: $(length(ig.tunnel)) x $(typeof(ig.tunnel))")
    println(io, " - ","values: $(typeof(ig.values)) -> $(isnothing(ig.values) ? 0 : dictsize(ig.values)) elements")
end


