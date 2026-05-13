



mutable struct DenseMPS{L, T<:Union{Float64, ComplexF64}} <: AbstractMPS
    ts::CachedVector{MPSTensor}
    center::Vector{Int64}

    function DenseMPS{L,T}(ts::CachedVector{MPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(ts,ct)
    end

    function DenseMPS{L,t}(ts::Vector{T}, cache_limit::Union{Int,Nothing}=nothing) where T <: Union{MPSTensor,MPSTensor{R}} where {L,t,R}
        limit = something(cache_limit, _cache_memory_limit(MPSTensor))
        return new{L,t}(CachedVector{MPSTensor}(ts, limit),[1,L])
    end

    function DenseMPS(ts::Vector{T}, cache_limit::Union{Int,Nothing}=nothing) where T <: Union{MPSTensor,MPSTensor{R}} where R
        L = length(ts)
        t = eltype(ts[1].A)
        limit = something(cache_limit, _cache_memory_limit(MPSTensor))
        return new{L,t}(CachedVector{MPSTensor}(ts, limit),[1,L])
    end

    function DenseMPS{L,T}(ts::Vector{AbstractTensorMap}, cache_limit::Union{Int,Nothing}=nothing) where {L,T}
        limit = something(cache_limit, _cache_memory_limit(MPSTensor))
        return new{L,T}(CachedVector{MPSTensor}([MPSTensor(t) for t in ts], limit),[1,L])
    end

end

mutable struct AdjointMPS{L, T<:Union{Float64, ComplexF64}} <: AbstractMPS
    ts::CachedVector{AdjointMPSTensor}
    center::Vector{Int64}

    function AdjointMPS{L,T}(ts::CachedVector{AdjointMPSTensor},
        ct::Vector{Int64}) where {L,T}
        return new{L,T}(ts,ct)
    end

    function AdjointMPS{L,T}(ts::Vector{AdjointMPSTensor}, cache_limit::Union{Int,Nothing}=nothing) where {L,T}
        limit = something(cache_limit, _cache_memory_limit(AdjointMPSTensor))
        return new{L,T}(CachedVector{AdjointMPSTensor}(ts, limit),[1,length(ts)])
    end

    function AdjointMPS{L,T}(ts::Vector{AbstractTensorMap}, cache_limit::Union{Int,Nothing}=nothing) where {L,T}
        limit = something(cache_limit, _cache_memory_limit(AdjointMPSTensor))
        return new{L,T}(CachedVector{AdjointMPSTensor}([AdjointMPSTensor(elm) for elm in ts], limit),[1,length(ts)])
    end

end

Base.adjoint(A::DenseMPS{L,T}) where {L,T} = AdjointMPS{L,T}(adjoint(A.ts), deepcopy(A.center))
Base.adjoint(A::AdjointMPS{L,T}) where {L,T} = DenseMPS{L,T}(adjoint(A.ts), deepcopy(A.center))







