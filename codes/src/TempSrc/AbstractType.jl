




abstract type AbstractMPSTensor{R} end
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
mutable struct MPSTensor{R} <: AbstractMPSTensor{R}
    A::AbstractTensorMap

    function MPSTensor(ts::AbstractTensorMap)
        return new{rank(ts)}(ts)
    end

    function MPSTensor{r}(ts::TrivialTensorMap) where r
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
mutable struct AdjointMPSTensor{R} <: AbstractMPSTensor{R}
    A::AbstractTensorMap

    function AdjointMPSTensor(ts::AbstractTensorMap)
        return new{rank(ts)}(ts)
    end

    function AdjointMPSTensor(fc::Function,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(fc,codomain,domain)
        return new{rank(A)}(A)
    end

end

function Base.adjoint(t::MPSTensor)
    return AdjointMPSTensor(t.A')
end

function Base.adjoint(ts::Vector{MPSTensor})
    return convert(Vector{AdjointMPSTensor},[AdjointMPSTensor(t.A') for t in ts])
end

function Base.adjoint(t::AdjointMPSTensor)
    return MPSTensor(t.A')
end

function Base.adjoint(ts::Vector{AdjointMPSTensor})
    return convert(Vector{MPSTensor},[MPSTensor(t.A') for t in ts])
end


function Base.:*(A::MPSTensor{3}, Ad::AdjointMPSTensor{3})
    return @tensor A.A[1,2,3] * Ad.A[3,1,2]
end

function Base.:*(n::Number, A::MPSTensor)
    return MPSTensor(A.A*n)
end

function Base.:*(n::Number, A::MPSTensor)
    return MPSTensor(A.A*n)
end

function Base.:/(A::MPSTensor, n::Number)
    @assert n ≠ 0
    return (1/n)*A
end

function Base.:+(A::MPSTensor{R₁}, B::MPSTensor{R₂}) where {R₁,R₂}
    @assert R₁ == R₂
    return MPSTensor(A.A + B.A)
end

function Base.:-(A::MPSTensor{R₁}, B::MPSTensor{R₂}) where {R₁,R₂}
    return A + (-1)*B
end

function TensorKit.norm(A::MPSTensor)
    return norm(A.A)
end
"""
todo {}
    1' - A - R
         | \
         2' 3'...(R-1)'
"""
mutable struct CompositeMPSTensor{N, R} <: AbstractMPSTensor{R}
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

function composite(A::MPSTensor{3}, B::MPSTensor{3})
    @tensor tmp[-1 -2 -3; -4] ≔ A.A[-1,-2,1]*B.A[1,-3,-4]
    return CompositeMPSTensor(tmp)
end

mutable struct AdjointCompositeMPSTensor{N, R} <: AbstractMPSTensor{R}
    A::AbstractTensorMap

    function AdjointCompositeMPSTensor(A::AbstractTensorMap)
        return new{length(domain(A))-1, rank(A)}(A)
    end

    function AdjointCompositeMPSTensor(fc::Function,codom,dom)
        A = TensorMap(fc,codom,dom)
        return new{length(domain(A))-1, rank(A)}(A)
    end
end


abstract type AbstractMPOTensor end

mutable struct DenseMPOTensor{R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function DenseMPOTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
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
end

function composite(A::DenseMPOTensor{4}, B::DenseMPOTensor{4})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ A.A[-2,-3,1,-6] * B.A[-1,1,-4,-5]
    return CompositeMPOTensor(tmp)
end

function Base.adjoint(t::CompositeMPSTensor)
    return AdjointCompositeMPSTensor(t.A')
end

function Base.adjoint(ts::Vector{CompositeMPSTensor})
    return convert(Vector{AdjointCompositeMPSTensor},[AdjointCompositeMPSTensor(t.A') for t in ts])
end

function Base.adjoint(t::AdjointCompositeMPSTensor)
    return CompositeMPSTensor(t.A')
end

function Base.adjoint(ts::Vector{AdjointCompositeMPSTensor})
    return convert(Vector{CompositeMPSTensor},[CompositeMPSTensor(t.A') for t in ts])
end

function Base.:*(A::CompositeMPSTensor{2, 4}, B::AdjointCompositeMPSTensor{2, 4})
    return @tensor A.A[1,2,3,4] * B.A[4,1,2,3]
end

function Base.:+(A::CompositeMPSTensor{2, 4}, B::CompositeMPSTensor{2, 4})
    return CompositeMPSTensor(A.A + B.A)
end

function Base.:-(A::CompositeMPSTensor{2, 4}, B::CompositeMPSTensor{2, 4})
    return A + (-1)*B
end

function Base.:*(n::Number, A::CompositeMPSTensor)
    return CompositeMPSTensor(n*A.A)
end

function Base.:/(A::CompositeMPSTensor, n::Number)
    return (1/n) * A
end

function Base.iterate(t::Union{AbstractMPSTensor, AbstractMPOTensor})
    return (t.A,nothing)
end

function Base.iterate(::Union{AbstractMPSTensor, AbstractMPOTensor},::Nothing)
    return nothing
end

function TensorKit.norm(A::Union{AbstractMPSTensor, AbstractMPOTensor})
    return norm(A.A)
end


function getAuxSpace(t::DenseMPOTensor{4})
    return collect(codomain(t.A))[2], collect(domain(t.A))[1]
end

function getAuxSpace(t::AdjointMPOTensor{4})
    return collect(domain(t.A))[2], collect(codomain(t.A))[1]
end


function Base.adjoint(t::DenseMPOTensor)
    return AdjointMPOTensor(t.A')
end

function Base.adjoint(ts::Vector{DenseMPOTensor})
    return convert(Vector{AdjointMPOTensor},[AdjointMPOTensor(t.A') for t in ts])
end

function Base.adjoint(t::AdjointMPOTensor)
    return DenseMPOTensor(t.A')
end

function Base.adjoint(ts::Vector{AdjointMPOTensor})
    return convert(Vector{DenseMPOTensor},[DenseMPOTensor(t.A') for t in ts])
end

mutable struct CompositeMPOTensor{N, R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function CompositeMPOTensor(A::AbstractTensorMap)
        return new{length(codomain(A))-1, rank(A)}(A)
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

    function AdjointCompositeMPOTensor(fc::Function,codom,dom)
        A = TensorMap(fc,codom,dom)
        return new{length(domain(A))-1, rank(A)}(A)
    end
end



mutable struct SparseMPOTensor{N,M} <: AbstractMPOTensor
    m::Matrix{Union{Nothing,DenseMPOTensor}}

    function SparseMPOTensor(m::Matrix{Union{Nothing,DenseMPOTensor}})
        return new{size(m)...}(m::Matrix{Union{Nothing,DenseMPOTensor}})
    end

    function SparseMPOTensor(::Nothing,N::Int64,M::Int64)
        return new{N,M}(Matrix{Union{Nothing,DenseMPOTensor}}(nothing,N,M))
    end
end

function Base.size(::SparseMPOTensor{N,M}) where {N,M}
    return N,M
end


function Base.length(::DenseMPOTensor)
    return 1
end

function Base.iterate(t::DenseMPOTensor)
    return (t,nothing)
end

function Base.iterate(::DenseMPOTensor,::Nothing)
    return nothing
end

abstract type AbstractMPS end
abstract type AbstractMPO end


function rank(A::AbstractTensorMap)
    return length(codomain(A)) + length(domain(A))
end


