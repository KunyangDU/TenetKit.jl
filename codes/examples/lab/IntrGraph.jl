using TensorKit
include("../../src/TensorWrapper/AbstractType.jl")
include("../../src/IntrTree/LocalOperator.jl")

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
        ops = ntuple(i -> LocalOperator(As[i], names[i], sites[i], i == N ? strength : 1.0), N)
        new{L,N}(ops, fermionic, Z)
    end
end

mutable struct InteractionGraph{L}
    graph::Any
    tunnel::Vector{<:InteractionTunnel{L}}
    values::Union{Nothing,Dict}
    InteractionGraph(tunnel::Vector{<:InteractionTunnel{L}}) where L = new{L}(nothing, tunnel, nothing)
end

import Base: getindex, length


function getindex(obj::InteractionTunnel{L,N}, i::Int64) where {L,N}
    sites = map(x -> x.site, obj.A)
    idx = findfirst(x -> x == i, sites)
    idx !== nothing && return obj.A[idx]
    isnothing(obj.Z) && return IdentityOperator(i)
    return iseven(sum([(sites[j] > i && obj.fermionic[j]) ? 1 : 0 for j in 1:N])) ? IdentityOperator(i) : LocalOperator(obj.Z, "Z", i)
end

length(::InteractionTunnel{L}) where L = L

function getindex(obj::InteractionTunnel, r::UnitRange{Int64})
    return [obj[i] for i in r]
end


