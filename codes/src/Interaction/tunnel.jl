
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
    function InteractionTunnel{L,T}(A::NTuple{N,T}, fermionic::NTuple{N,Bool}, Z::Union{Nothing,AbstractTensorMap}, strength::Float64) where {L,T,N}
        return new{L,T,N}(A,fermionic,Z,strength)
    end

    function InteractionTunnel(
        A::AbstractTensorMap,
        site::Int64,
        name::String,
        fermionic::Bool,
        strength::Number,
        Z::Union{Nothing,AbstractTensorMap},
        L::Int64,T::Type = LocalOperator
    )
        return new{L,T,1}((T(A,name,site),),(fermionic,),Z,strength)
    end

    function InteractionTunnel(
        A::NTuple{N,T},
        fermionic::NTuple{N,Bool},
        Z::Union{Nothing,AbstractTensorMap},
        strength::Float64, L::Int64,T′::Type = T
    ) where {N,T}
        @assert T <: T′
        return new{L,T′,N}(A,fermionic,Z,strength)
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

Base.getindex(obj::CompositeInteractionTunnel, i::Int64) = CompositeLocalOperator([a[i] for a in obj.A])
composite(A::InteractionTunnel{L,T}, B::InteractionTunnel{L,T}) where {L,T <: LocalOperator} = CompositeInteractionTunnel{L,CompositeLocalOperator{2},2}(NTuple{2,InteractionTunnel{L}}((A,B)), A.strength * B.strength)


Base.getindex(obj::AbstractTunnel, r::UnitRange{Int64}) = [obj[i] for i in r]
Base.length(::AbstractTunnel{L}) where L = L

_site(A::InteractionTunnel) = map(x -> x.site,A.A)
Base.:*(A::Number, B::InteractionTunnel) = (B′ = deepcopy(B); B′.strength *= A; return B′)
