using TensorKit
abstract type AbstractGraph end

mutable struct SimpleGraph{N,T} <: AbstractGraph
    node::NTuple{N,T}
    function SimpleGraph(node::Tuple)
        N = length(node)
        T = mapreduce(typeof, typejoin, node)
        return new{N,T}(node)
    end
end

mutable struct InteractionTunnel{L,N}
    A::NTuple{N,LocalOperator}
    fermionic::NTuple{N,Bool}
    Z::Union{Nothing,AbstractTensorMap}
    function InteractionTunnel(
        As::NTuple{N,AbstractTensorMap},
        sites::NTuple{N,Int64},
        names::NTuple{N,String},
        fermionic::NTuple{N,Bool},
        strength::Number,
        Z::Union{Nothing,AbstractTensorMap},
        L::Int64
    ) where N
        # ops = ntuple(i -> LocalOperator(As[i], names[i], sites[i], i == N ? strength : 1.0), N)
        ops = ntuple(i -> LocalOperator(As[i], names[i], sites[i], 1.0), N)
        new{L,N}(ops, fermionic, Z)
    end
end

mutable struct InteractionGraph{L,G<:AbstractGraph}
    graph::Union{Nothing, G}
    tunnel::Vector{<:InteractionTunnel{L}}
    values::Union{Nothing, Dict}
    L::Int64

    function InteractionGraph(tunnel::Vector{<:InteractionTunnel{L}}) where L
        return new{L, SimpleGraph}(nothing, tunnel, nothing, L)
    end

    function InteractionGraph(L::Int64)
        return new{L, SimpleGraph}(nothing, Vector{InteractionTunnel{L}}(), nothing, L)
    end
end

function Base.getindex(obj::InteractionTunnel{L,N}, i::Int64) where {L,N}
    sites = map(x -> x.site, obj.A)
    idx = findfirst(x -> x == i, sites)
    idx !== nothing && return obj.A[idx]
    isnothing(obj.Z) && return IdentityOperator(i)
    return iseven(sum([(sites[j] > i && obj.fermionic[j]) ? 1 : 0 for j in 1:N])) ? IdentityOperator(i) : LocalOperator(obj.Z, "Z", i)
end

Base.length(::InteractionTunnel{L}) where L = L

function Base.getindex(obj::InteractionTunnel, r::UnitRange{Int64})
    return [obj[i] for i in r]
end

# -- 图构建：从 tunnels 建立 DAG --
function build_graph!(ig::InteractionGraph{L}) where L
    isempty(ig.tunnel) && return ig
    left_root, right_root = build_intrmap(ig.tunnel)
    minimize!(left_root)
    ig.graph = SimpleGraph((left_root, right_root))
    return ig
end

