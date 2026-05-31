
mutable struct InteractionTunnel{L,T,N} <: AbstractTunnel{L,T}
    A::NTuple{N,T}
    fermionic::NTuple{N,Bool}
    Z::Union{Nothing,AbstractTensorMap}
    strength::Float64
    function InteractionTunnel(
        As::NTuple{N,AbstractTensorMap},
        sites::NTuple{N,Int64},
        names::NTuple{N,String},
        fermionic::NTuple{N,Bool},
        strength::Number,
        Z::Union{Nothing,AbstractTensorMap},
        L::Int64,T::Type = LocalOperator
    ) where N
        ops = ntuple(i -> T(As[i], names[i], sites[i]), N)
        new{L,T,N}(ops, fermionic, Z, Float64(strength))
    end
end

function Base.getindex(obj::InteractionTunnel{L,<:LocalOperator,N}, i::Int64) where {L,N}
    sites = map(x -> x.site, obj.A)
    idx = findfirst(x -> x == i, sites)
    idx !== nothing && return obj.A[idx]
    isnothing(obj.Z) && return IdentityOperator(i)
    return iseven(sum([(sites[j] > i && obj.fermionic[j]) ? 1 : 0 for j in 1:N])) ? IdentityOperator(i) : LocalOperator(obj.Z, "Z", i)
end

mutable struct CompositeInteractionTunnel{L,T,N} <: AbstractTunnel{L,T}
    A::NTuple{N,InteractionTunnel{L}}
    strength::Float64
end

Base.getindex(obj::CompositeInteractionTunnel, i::Int64) = CompositeObservableOperator([a[i] for a in obj.A])
composite(A::InteractionTunnel{L,T}, B::InteractionTunnel{L,T}) where {L,T <: ObservableOperator} = CompositeInteractionTunnel{L,CompositeObservableOperator{2},2}(NTuple{2,InteractionTunnel{L}}((A,B)), A.strength * B.strength)


Base.getindex(obj::AbstractTunnel, r::UnitRange{Int64}) = [obj[i] for i in r]
Base.length(::AbstractTunnel{L}) where L = L

