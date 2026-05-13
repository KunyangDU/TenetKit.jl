


mutable struct SparseMPO{L} <: AbstractMPO
    ts::Vector{SparseMPOTensor}
    D::Vector{NTuple{2,Int64}}
    
    function SparseMPO(ts::Vector{SparseMPOTensor},
        D::Vector{NTuple{2,Int64}})
        return new{length(ts)}(ts,D)
    end

    function SparseMPO(ts::Vector{T}) where T <: Union{SparseMPOTensor,SparseMPOTensor{N,M}} where {N,M}
        D = map(size,ts)
        return new{length(ts)}(ts,convert(Vector{NTuple{2,Int64}},D))
    end

    function SparseMPO(t::SparseMPOTensor{N,M}) where {N,M}
        D = convert(Vector{NTuple{2,Int64}},[(N,M)])
        ts = convert(Vector{SparseMPOTensor},[t])
        return new{length(ts)}(ts,D)        
    end
end

mutable struct DenseMPO{L} <: AbstractMPO
    ts::CachedVector{DenseMPOTensor}
    center::Vector{Int64}

    function DenseMPO(A::CachedVector{DenseMPOTensor},center::Vector{Int64})
        return new{length(A)}(A,center)
    end

    function DenseMPO(A::Vector{DenseMPOTensor{R}}, cache_limit::Union{Int,Nothing}=nothing) where R
        limit = something(cache_limit, _cache_memory_limit(DenseMPOTensor))
        return new{length(A)}(CachedVector{DenseMPOTensor}(A, limit),[1,length(A)])
    end

    function DenseMPO(t::DenseMPOTensor, cache_limit::Union{Int,Nothing}=nothing)
        limit = something(cache_limit, _cache_memory_limit(DenseMPOTensor))
        A = CachedVector{DenseMPOTensor}([t], limit)
        return new{1}(A,[1,1])
    end

    function DenseMPO(t::Vector, cache_limit::Union{Int,Nothing}=nothing)
        limit = something(cache_limit, _cache_memory_limit(DenseMPOTensor))
        tmp = map(DenseMPOTensor,t)
        A = CachedVector{DenseMPOTensor}(tmp, limit)
        return new{length(A)}(A,[1,length(A)])
    end
end
const DenseMPQ = Union{DenseMPO,DenseMPS}

mutable struct AdjointMPO{L} <: AbstractMPO
    ts::CachedVector{AdjointMPOTensor}
    center::Vector{Int64}

    function AdjointMPO(A::CachedVector{AdjointMPOTensor},center::Vector{Int64})
        return new{length(A)}(A,center)
    end

    function AdjointMPO(A::Vector{AdjointMPOTensor{R}}, cache_limit::Union{Int,Nothing}=nothing) where R
        limit = something(cache_limit, _cache_memory_limit(AdjointMPOTensor))
        return new{length(A)}(CachedVector{AdjointMPOTensor}(A, limit),[1,length(A)])
    end

    function AdjointMPO(t::AdjointMPOTensor, cache_limit::Union{Int,Nothing}=nothing)
        limit = something(cache_limit, _cache_memory_limit(AdjointMPOTensor))
        A = CachedVector{AdjointMPOTensor}([t], limit)
        return new{1}(A,[1,1])
    end

    function AdjointMPO(t::Vector{AbstractTensorMap}, cache_limit::Union{Int,Nothing}=nothing)
        limit = something(cache_limit, _cache_memory_limit(AdjointMPOTensor))
        tmp = map(AdjointMPOTensor,t)
        A = CachedVector{AdjointMPOTensor}(tmp, limit)
        return new{length(A)}(A,[1,length(A)])
    end
end

Base.adjoint(A::DenseMPO{L}) where {L} = AdjointMPO(adjoint(A.ts), deepcopy(A.center))
Base.adjoint(A::AdjointMPO{L}) where {L} = DenseMPO(adjoint(A.ts), deepcopy(A.center))

isadjoint(::DenseMPO) = false
isadjoint(::AdjointMPO) = true
isref(::DenseMPO) = false
isref(::AdjointMPO) = false

mutable struct RefMPO{L} <: AbstractMPO
    ts::CachedVector{DenseMPOTensor}
    center::Vector{Int64}
    mapping::Function
    pointer::DenseMPO
    RefMPO(A::DenseMPO{L},mapping::Function = identity) where L = new{L}(A.ts,A.center,mapping,A)
end

issparse(::RefMPO) = false
isadjoint(::RefMPO) = false
isref(::RefMPO) = true

