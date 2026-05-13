


mutable struct LocalEnvironmentTensor{R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LocalEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
    function LocalEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{rank(t)}(t)
    end
end

mutable struct RightEnvironmentTensor{R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function RightEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
    function RightEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{rank(t)}(t)
    end
end

mutable struct LeftEnvironmentTensor{R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LeftEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
    function LeftEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{rank(t)}(t)
    end
end

Base.adjoint(A::LeftEnvironmentTensor{2}) = LeftEnvironmentTensor{2}(A.A')
Base.adjoint(A::RightEnvironmentTensor{2}) = RightEnvironmentTensor{2}(A.A')

# function Base.adjoint(A::LeftEnvironmentTensor{3})
#     tmp = A.A'
#     LeftEnvironmentTensor{3}(permute(tmp,(2,),(1,3)))
# end

# function Base.adjoint(A::RightEnvironmentTensor{3})
#     tmp = A.A'
#     RightEnvironmentTensor{3}(permute(tmp,(1,),(3,2)))
# end

"""

"""
mutable struct LeftCompositeEnvironmentTensor{Rcd,Rt,N,I} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LeftCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{length(codomain(t)),rank(t),3,3}(t)
    end

    function LeftCompositeEnvironmentTensor{n,r}(t::AbstractTensorMap) where {n,r}
        return new{length(codomain(t)),rank(t),3,3}(t)
    end

    function LeftCompositeEnvironmentTensor(t::AbstractTensorMap,n::Int64,i::Int64)
        return new{length(codomain(t)),rank(t),n,i}(t)
    end

    function LeftCompositeEnvironmentTensor{rcd,rt,n,i}(t::AbstractTensorMap) where {rcd,rt,n,i}
        return new{rcd,rt,n,i}(t)
    end
end

mutable struct RightCompositeEnvironmentTensor{Rcd,Rt,N,I} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function RightCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{length(domain(t)),rank(t),3,3}(t)
    end

    function RightCompositeEnvironmentTensor{n,r}(t::AbstractTensorMap) where {n,r}
        return new{length(domain(t)),rank(t),3,3}(t)
    end
    function RightCompositeEnvironmentTensor(t::AbstractTensorMap,n::Int64,i::Int64)
        return new{length(domain(t)),rank(t),n,i}(t)
    end
    function RightCompositeEnvironmentTensor{rcd,rt,n,i}(t::AbstractTensorMap) where {rcd,rt,n,i}
        return new{rcd,rt,n,i}(t)
    end
end

mutable struct SparseEnvironmentTensor <: AbstractEnvironmentTensor
    A::Vector{AbstractEnvironmentTensor}
    D::Int64

    function SparseEnvironmentTensor(t::Vector{AbstractEnvironmentTensor},D::Int64)
        return new(t,D)
    end

    function SparseEnvironmentTensor(t::Vector{AbstractEnvironmentTensor})
        return new(t,length(t))
    end

    function SparseEnvironmentTensor(t::Union{LeftEnvironmentTensor,RightEnvironmentTensor})
        return new(convert(Vector{AbstractEnvironmentTensor},[t]),1)
    end
end



mutable struct SparseLeftEnvironmentTensor{N} <: AbstractLeftEnvironmentTensor
    A::Union{Array{LeftEnvironmentTensor},Array{LeftCompositeEnvironmentTensor}}
    D::NTuple{N,Int}

    function SparseLeftEnvironmentTensor(t::Array{LeftEnvironmentTensor},D::Int64)
        return new{1}(t,(D,))
    end

    function SparseLeftEnvironmentTensor(t::Union{Array{LeftEnvironmentTensor},Array{LeftCompositeEnvironmentTensor}})
        return new{ndims(t)}(t,size(t))
    end

    function SparseLeftEnvironmentTensor(t::LeftEnvironmentTensor)
        return new{1}(convert(Array{LeftEnvironmentTensor},[t]),(1,))
    end

    function SparseLeftEnvironmentTensor(t::AbstractTensorMap)
        return new{1}(convert(Array{LeftEnvironmentTensor},[LeftEnvironmentTensor(t)]),(1,))
    end

    function SparseLeftEnvironmentTensor(t::Array)
        return new{ndims(t)}(convert(Array{LeftEnvironmentTensor},[LeftEnvironmentTensor(ti) for ti in t]),size(t))
    end
end

mutable struct SparseRightEnvironmentTensor{N} <: AbstractRightEnvironmentTensor
    A::Union{Array{RightEnvironmentTensor},Array{RightCompositeEnvironmentTensor}}
    D::NTuple{N,Int}

    function SparseRightEnvironmentTensor(t::Array{RightEnvironmentTensor},D::Int64)
        return new{1}(t,(D,))
    end

    function SparseRightEnvironmentTensor(t::Union{Array{RightEnvironmentTensor},Array{RightCompositeEnvironmentTensor}})
        return new{ndims(t)}(t,size(t))
    end

    function SparseRightEnvironmentTensor(t::RightEnvironmentTensor)
        return new{1}(convert(Array{RightEnvironmentTensor},[t]),(1,))
    end

    function SparseRightEnvironmentTensor(t::AbstractTensorMap)
        return new{1}(convert(Array{RightEnvironmentTensor},[RightEnvironmentTensor(t)]),(1,))
    end

    function SparseRightEnvironmentTensor(t::Array)
        return new{ndims(t)}(convert(Array{RightEnvironmentTensor},[RightEnvironmentTensor(ti) for ti in t]),size(t))
    end
end


mutable struct DenseLeftEnvironmentTensor{R} <: AbstractLeftEnvironmentTensor
    A::LeftEnvironmentTensor

    function DenseLeftEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(LeftEnvironmentTensor(t))
    end

    function DenseLeftEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{rank(t)}(LeftEnvironmentTensor(t))
    end

    function DenseLeftEnvironmentTensor(t::LeftEnvironmentTensor)
        return new{rank(t.A)}(t)
    end
end

mutable struct DenseRightEnvironmentTensor{R} <: AbstractLeftEnvironmentTensor
    A::RightEnvironmentTensor

    function DenseRightEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(RightEnvironmentTensor(t))
    end

    function DenseRightEnvironmentTensor{r}(t::AbstractTensorMap) where r
        return new{rank(t)}(RightEnvironmentTensor(t))
    end

    function DenseRightEnvironmentTensor(t::RightEnvironmentTensor)
        return new{rank(t.A)}(t)
    end
end

Base.adjoint(A::DenseLeftEnvironmentTensor{2}) = DenseLeftEnvironmentTensor(A.A')
Base.adjoint(A::DenseRightEnvironmentTensor{2}) = DenseRightEnvironmentTensor(A.A')

"""
Monolayer Environment, i.e., only one layer MPO is considered.
"""
mutable struct Environment{N,L} <: AbstractEnvironment
    layer::Vector
    envs::Union{Nothing,CachedVector{AbstractEnvironmentTensor}}
    center::Vector{Int64}
    L::Int64

    function Environment(layer::Vector,
        envs::CachedVector{AbstractEnvironmentTensor},
        center::Union{Nothing,Vector{Int64}},
        L::Union{Nothing,Int64})
        return new{length(layer),length(layer[1])}(layer,envs,center,L)
    end

    function Environment(layer::Vector)
        L = length(layer[1])
        return new{length(layer),length(layer[1])}(layer,nothing,[1,L],L)
    end

end

mutable struct CBEenvironment <: AbstractEnvironment
    tL₀::AbstractTensorWrapper
    tR₀::AbstractTensorWrapper
    tL::Union{AbstractTensorWrapper,Nothing}
    tR::Union{AbstractTensorWrapper,Nothing}
    D_i::Int64
    D_f::Int64
    Λ::Union{AbstractTensorWrapper,Nothing}
    Lorth::Union{SparseLeftEnvironmentTensor,LeftCompositeEnvironmentTensor,LeftEnvironmentTensor,DenseLeftEnvironmentTensor,Nothing}
    Rorth::Union{SparseRightEnvironmentTensor,RightCompositeEnvironmentTensor,RightEnvironmentTensor,DenseRightEnvironmentTensor,Nothing}

    # function CBEenvironment(env::Environment{N,L},ind::Int64,D_i::Int64,D_f::Int64) where {N,L}
    #     return new(env,nothing,nothing,nothing,ind,D_i,D_f,N,L)
    # end

    # function CBEenvironment(
    #     Λ::AbstractTensorWrapper,
    #     tl₀::AbstractTensorWrapper,
    #     tr₀::AbstractTensorWrapper,
    #     tl::AbstractTensorWrapper,
    #     tr::AbstractTensorWrapper,
    #     Lorth::LeftCompositeEnvironmentTensor,
    #     Rorth::RightCompositeEnvironmentTensor,
    #     D_i::Int64,
    #     D_f::Int64,
    # )
    # return new(Λ,tl₀,tr₀,tl,tr,Lorth,Rorth,D_i,D_f)
    # end
end

# function _fullCBEenv(tL₀::AbstractTensorWrapper,tR₀::AbstractTensorWrapper,D_i::Int64,D_f::Int64)
#     return CBEenvironment(tL₀,tR₀,nothing,nothing,D_i,D_f,nothing,nothing,nothing)
# end

# ───────────────────────────────────────────────────────────────
# Per-type cache memory limit for Environment tensors
# ───────────────────────────────────────────────────────────────

_cache_memory_limit(::Type{<:AbstractEnvironmentTensor}) = round(Int, CACHE_MEMORY_LIMIT[] * ENV_CACHE_RATIO[])

# function _randCBEenv(
#     tL₀::AbstractTensorWrapper,tR₀::AbstractTensorWrapper,
#     tL::Union{AbstractTensorWrapper,Nothing},tR::Union{AbstractTensorWrapper,Nothing},
#     D_i::Int64,D_f::Int64,Λ::AbstractTensorWrapper,
#     Lorth::Union{SparseLeftEnvironmentTensor,LeftCompositeEnvironmentTensor,LeftEnvironmentTensor},
#     Rorth::Union{SparseRightEnvironmentTensor,RightCompositeEnvironmentTensor,RightEnvironmentTensor})
#     return CBEenvironment(tL₀,tR₀,tL,tR,D_i,D_f,Λ,Lorth,Rorth)
# end

