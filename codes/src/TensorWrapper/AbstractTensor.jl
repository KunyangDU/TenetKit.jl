
"""
     struct MPSTensor <: AbstractMPSTensor
          A::AbstractTensorMap
     end 
          
Wrapper type for MPS local tensors.

Convention (' marks codomain): 

    1' - A - R
         | \
         2' 3...R-1

In particular, R == 2 for bond tensor.

# Constructors
     MPSTensor(::AbstractTensorMap) 
"""
mutable struct MPSTensor{R} <: AbstractMPSTensor 
    A::AbstractTensorMap

    function MPSTensor(ts::AbstractTensorMap)
        return new{rank(ts)}(ts)
    end

    function MPSTensor{r}(ts::AbstractTensorMap) where r
        return new{r}(ts)
    end

    function MPSTensor(fc::Function,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(fc,codomain,domain)
        return new{rank(A)}(A)
    end

    function MPSTensor(data::AbstractMatrix,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(data[:],codomain,domain)
        return new{rank(A)}(A)
    end

    function MPSTensor(data::AbstractVector,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(data,codomain,domain)
        return new{rank(A)}(A)
    end

end
"""
Wrapper type for ajoint of MPS local tensors.

Convention (' marks codomain): 

    1 - A - R'
        | \
        2 3'...(R-1)'
         
In particular, R == 2 for bond tensor.

# Constructors
     MPSTensor(::AbstractTensorMap) 
"""
mutable struct AdjointMPSTensor{R} <: AbstractMPSTensor
    A::AbstractTensorMap

    function AdjointMPSTensor(ts::AbstractTensorMap)
        return new{rank(ts)}(ts)
    end

    function AdjointMPSTensor(fc::Function,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(fc,codomain,domain)
        return new{rank(A)}(A)
    end

end

Base.adjoint(t::MPSTensor) = AdjointMPSTensor(t.A')
Base.adjoint(ts::Vector{MPSTensor}) = convert(Vector{AdjointMPSTensor},[AdjointMPSTensor(t.A') for t in ts])
Base.adjoint(ts::AbstractVector{MPSTensor}) = convert(Vector{AdjointMPSTensor},[AdjointMPSTensor(t.A') for t in ts])
Base.adjoint(t::AdjointMPSTensor) = MPSTensor(t.A')
Base.adjoint(ts::Vector{AdjointMPSTensor}) = convert(Vector{MPSTensor},[MPSTensor(t.A') for t in ts])
Base.adjoint(ts::AbstractVector{AdjointMPSTensor}) = convert(Vector{MPSTensor},[MPSTensor(t.A') for t in ts])

"""
todo {}
    1' - A - R
         | \
         2' 3'...(R-1)'
"""
mutable struct CompositeMPSTensor{N, R} <: AbstractMPSTensor
    A::AbstractTensorMap

    function CompositeMPSTensor(A::AbstractTensorMap)
        return new{length(codomain(A))-1, rank(A)}(A)
    end
    function CompositeMPSTensor{n,r}(A::AbstractTensorMap) where {n,r}
        return new{n,r}(A)
    end

    function CompositeMPSTensor(fc::Function, codom, dom)
        A = TensorMap(fc,codom,dom)
        return new{length(codomain(A))-1, rank(A)}(A)
    end
end

mutable struct AdjointCompositeMPSTensor{N, R} <: AbstractMPSTensor
    A::AbstractTensorMap

    function AdjointCompositeMPSTensor(A::AbstractTensorMap)
        return new{length(domain(A))-1, rank(A)}(A)
    end

    function AdjointCompositeMPSTensor{n,r}(A::AbstractTensorMap) where {n,r}
        return new{n, r}(A)
    end

    function AdjointCompositeMPSTensor(fc::Function,codom,dom)
        A = TensorMap(fc,codom,dom)
        return new{length(domain(A))-1, rank(A)}(A)
    end
end

Base.adjoint(t::CompositeMPSTensor) = AdjointCompositeMPSTensor(t.A')
Base.adjoint(t::AdjointCompositeMPSTensor) = CompositeMPSTensor(t.A')
Base.adjoint(ts::Vector{CompositeMPSTensor}) = convert(Vector{AdjointCompositeMPSTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{CompositeMPSTensor}) = convert(Vector{AdjointCompositeMPSTensor},[t' for t in ts])
Base.adjoint(ts::Vector{AdjointCompositeMPSTensor}) = convert(Vector{CompositeMPSTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{AdjointCompositeMPSTensor}) = convert(Vector{CompositeMPSTensor},[t' for t in ts])

mutable struct DenseMPOTensor{R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function DenseMPOTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end

    function DenseMPOTensor{r}(t::AbstractTensorMap) where r
        return new{r}(t)
    end
    function DenseMPOTensor(fc::Function,codom,dom)
        A = TensorMap(fc,codom,dom)
        return new{rank(A)}(A)
    end
end


mutable struct AdjointMPOTensor{R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function AdjointMPOTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end

    function AdjointMPOTensor(fc::Function,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(fc,codomain,domain)
        return new{rank(A)}(A)
    end

    function AdjointMPOTensor{r}(t::AbstractTensorMap) where r
        return new{r}(t)
    end
end

Base.adjoint(t::DenseMPOTensor) = AdjointMPOTensor(t.A')
Base.adjoint(t::AdjointMPOTensor) = DenseMPOTensor(t.A')
Base.adjoint(ts::Vector{DenseMPOTensor}) = convert(Vector{AdjointMPOTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{DenseMPOTensor}) = convert(Vector{AdjointMPOTensor},[t' for t in ts])
Base.adjoint(ts::Vector{AdjointMPOTensor}) = convert(Vector{DenseMPOTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{AdjointMPOTensor}) = convert(Vector{DenseMPOTensor},[t' for t in ts])

mutable struct CompositeMPOTensor{N, R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function CompositeMPOTensor(A::AbstractTensorMap)
        return new{length(codomain(A))-1, rank(A)}(A)
    end

    function CompositeMPOTensor{n,r}(A::AbstractTensorMap) where {n,r}
        return new{n,r}(A)
    end

    function CompositeMPOTensor(fc::Function, codom, dom)
        A = TensorMap(fc,codom,dom)
        return new{length(codomain(A))-1, rank(A)}(A)
    end
end

mutable struct AdjointCompositeMPOTensor{N, R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function AdjointCompositeMPOTensor(A::AbstractTensorMap)
        return new{length(domain(A))-1, rank(A)}(A)
    end

    function AdjointCompositeMPOTensor{n,r}(A::AbstractTensorMap) where {n,r}
        return new{n,r}(A)
    end

    function AdjointCompositeMPOTensor(fc::Function,codom,dom)
        A = TensorMap(fc,codom,dom)
        return new{length(domain(A))-1, rank(A)}(A)
    end
end

Base.adjoint(t::CompositeMPOTensor) = return AdjointCompositeMPOTensor(t.A')
Base.adjoint(t::AdjointCompositeMPOTensor) = return CompositeMPOTensor(t.A')
Base.adjoint(ts::Vector{CompositeMPOTensor}) = return convert(Vector{AdjointCompositeMPOTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{CompositeMPOTensor}) = return convert(Vector{AdjointCompositeMPOTensor},[t' for t in ts])
Base.adjoint(ts::Vector{AdjointCompositeMPOTensor}) = convert(Vector{CompositeMPOTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{AdjointCompositeMPOTensor}) = convert(Vector{CompositeMPOTensor},[t' for t in ts])

mutable struct SparseMPOTensor{N,M} <: AbstractMPOTensor
    m::Matrix{Union{Nothing, AbstractLocalOperator}}

    function SparseMPOTensor(m::Matrix{Union{Nothing,AbstractLocalOperator}})
        return new{size(m)...}(m::Matrix{Union{Nothing,AbstractLocalOperator}})
    end

    function SparseMPOTensor(::Nothing,N::Int64,M::Int64)
        return new{N,M}(Matrix{Union{Nothing,AbstractLocalOperator}}(nothing,N,M))
    end
end


#= ====================== =#

# ───────────────────────────────────────────────────────────────
# Per-type cache memory limits (dispatch added after types defined)
# ───────────────────────────────────────────────────────────────

_cache_memory_limit(::Type{<:AbstractMPSTensor}) = round(Int, CACHE_MEMORY_LIMIT[] * MPS_CACHE_RATIO[])
_cache_memory_limit(::Type{<:AbstractMPOTensor}) = round(Int, CACHE_MEMORY_LIMIT[] * MPO_CACHE_RATIO[])

# ───────────────────────────────────────────────────────────────
# CachedVector adjoint methods — one-at-a-time iteration, returns CachedVector
# ───────────────────────────────────────────────────────────────

function Base.adjoint(cv::CachedVector{MPSTensor})
    result = CachedVector{AdjointMPSTensor}(length(cv), cache_capacity(cv))
    for i in 1:length(cv)
        result[i] = AdjointMPSTensor(cv[i].A')
    end
    return result
end

function Base.adjoint(cv::CachedVector{AdjointMPSTensor})
    result = CachedVector{MPSTensor}(length(cv), cache_capacity(cv))
    for i in 1:length(cv)
        result[i] = MPSTensor(cv[i].A')
    end
    return result
end

function Base.adjoint(cv::CachedVector{CompositeMPSTensor})
    result = CachedVector{AdjointCompositeMPSTensor}(length(cv), cache_capacity(cv))
    for i in 1:length(cv)
        result[i] = cv[i]'
    end
    return result
end

function Base.adjoint(cv::CachedVector{AdjointCompositeMPSTensor})
    result = CachedVector{CompositeMPSTensor}(length(cv), cache_capacity(cv))
    for i in 1:length(cv)
        result[i] = cv[i]'
    end
    return result
end

function Base.adjoint(cv::CachedVector{DenseMPOTensor})
    result = CachedVector{AdjointMPOTensor}(length(cv), cache_capacity(cv))
    for i in 1:length(cv)
        result[i] = cv[i]'
    end
    return result
end

function Base.adjoint(cv::CachedVector{AdjointMPOTensor})
    result = CachedVector{DenseMPOTensor}(length(cv), cache_capacity(cv))
    for i in 1:length(cv)
        result[i] = cv[i]'
    end
    return result
end

function Base.adjoint(cv::CachedVector{CompositeMPOTensor})
    result = CachedVector{AdjointCompositeMPOTensor}(length(cv), cache_capacity(cv))
    for i in 1:length(cv)
        result[i] = cv[i]'
    end
    return result
end

function Base.adjoint(cv::CachedVector{AdjointCompositeMPOTensor})
    result = CachedVector{CompositeMPOTensor}(length(cv), cache_capacity(cv))
    for i in 1:length(cv)
        result[i] = cv[i]'
    end
    return result
end
