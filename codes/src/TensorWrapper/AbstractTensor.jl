
"""
     struct MPSTensor{S<:Number, R, T<:AbstractTensorMap} <: AbstractMPSTensor
          A::T
     end

Wrapper type for MPS local tensors.  The first type parameter `S` is the scalar
(numeric) type shared by the TensorMap and its underlying data matrix.

Convention (' marks codomain):

    1' - A - R
         | \\
         2' 3...R-1

In particular, R == 2 for bond tensor.

# Constructors
     MPSTensor(::AbstractTensorMap)
     MPSTensor{S}(::AbstractTensorMap)   # override scalar type; R and TM inferred
"""
mutable struct MPSTensor{S<:Number, R, T<:AbstractTensorMap} <: AbstractMPSTensor
    A::T

    # Primary: infer everything from the tensor.
    function MPSTensor(ts::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(ts), rank(ts), TM}(ts)
    end

    # Explicitly override the scalar type S; rank and TM are still inferred.
    function MPSTensor{S}(ts::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, rank(ts), TM}(ts)
    end

    # Identity constructor for fully-parameterized type (needed by typeof(obj)(tm) dispatch).
    function MPSTensor{S, R, TM}(ts::TM) where {S<:Number, R, TM<:AbstractTensorMap}
        return new{S, R, TM}(ts)
    end

    function MPSTensor(fc::Function, codomain::Union{VectorSpace,ElementarySpace}, domain::Union{VectorSpace,ElementarySpace}; type::Type{<:Number} = scalartype(codomain))
        A = TensorMap(fc, type, codomain, domain)
        return new{scalartype(A), rank(A), typeof(A)}(A)
    end

    function MPSTensor(data::AbstractMatrix, codomain::Union{VectorSpace,ElementarySpace}, domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(data[:], codomain, domain)
        return new{scalartype(A), rank(A), typeof(A)}(A)
    end

    function MPSTensor(data::AbstractVector, codomain::Union{VectorSpace,ElementarySpace}, domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(data, codomain, domain)
        return new{scalartype(A), rank(A), typeof(A)}(A)
    end

end

"""
Wrapper type for adjoint of MPS local tensors.

Convention (' marks codomain):

    1 - A - R'
        | \\
        2 3'...(R-1)'

In particular, R == 2 for bond tensor.

# Constructors
     AdjointMPSTensor(::AbstractTensorMap)
     AdjointMPSTensor{S}(::AbstractTensorMap)
"""
mutable struct AdjointMPSTensor{S<:Number, R, T<:AbstractTensorMap} <: AbstractMPSTensor
    A::T

    function AdjointMPSTensor(ts::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(ts), rank(ts), TM}(ts)
    end

    function AdjointMPSTensor{S}(ts::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, rank(ts), TM}(ts)
    end

    function AdjointMPSTensor{S, R, TM}(ts::TM) where {S<:Number, R, TM<:AbstractTensorMap}
        return new{S, R, TM}(ts)
    end

    function AdjointMPSTensor(fc::Function, codomain::Union{VectorSpace,ElementarySpace}, domain::Union{VectorSpace,ElementarySpace}; type::Type{<:Number} = scalartype(codomain))
        A = TensorMap(fc, type, codomain, domain)
        return new{scalartype(A), rank(A), typeof(A)}(A)
    end

end

Base.adjoint(t::MPSTensor) = AdjointMPSTensor(t.A')
Base.adjoint(ts::Vector{<:MPSTensor}) = [AdjointMPSTensor(t.A') for t in ts]
Base.adjoint(t::AdjointMPSTensor) = MPSTensor(t.A')
Base.adjoint(ts::Vector{<:AdjointMPSTensor}) = [MPSTensor(t.A') for t in ts]

"""
todo {}
    1' - A - R
         | \\
         2' 3'...(R-1)'
"""
mutable struct CompositeMPSTensor{S<:Number, N, R, T<:AbstractTensorMap} <: AbstractMPSTensor
    A::T

    function CompositeMPSTensor(A::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(A), length(codomain(A))-1, rank(A), TM}(A)
    end

    function CompositeMPSTensor{S}(A::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, length(codomain(A))-1, rank(A), TM}(A)
    end

    function CompositeMPSTensor{S, N, R, TM}(A::TM) where {S<:Number, N, R, TM<:AbstractTensorMap}
        return new{S, N, R, TM}(A)
    end

    function CompositeMPSTensor(fc::Function, codom, dom; type::Type{<:Number} = scalartype(codom))
        A = TensorMap(fc, type, codom, dom)
        return new{scalartype(A), length(codomain(A))-1, rank(A), typeof(A)}(A)
    end
end

mutable struct AdjointCompositeMPSTensor{S<:Number, N, R, T<:AbstractTensorMap} <: AbstractMPSTensor
    A::T

    function AdjointCompositeMPSTensor(A::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(A), length(domain(A))-1, rank(A), TM}(A)
    end

    function AdjointCompositeMPSTensor{S}(A::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, length(domain(A))-1, rank(A), TM}(A)
    end

    function AdjointCompositeMPSTensor{S, N, R, TM}(A::TM) where {S<:Number, N, R, TM<:AbstractTensorMap}
        return new{S, N, R, TM}(A)
    end

    function AdjointCompositeMPSTensor(fc::Function, codom, dom; type::Type{<:Number} = scalartype(codom))
        A = TensorMap(fc, type, codom, dom)
        return new{scalartype(A), length(domain(A))-1, rank(A), typeof(A)}(A)
    end
end

Base.adjoint(t::CompositeMPSTensor) = AdjointCompositeMPSTensor(t.A')
Base.adjoint(t::AdjointCompositeMPSTensor) = CompositeMPSTensor(t.A')
Base.adjoint(ts::Vector{<:CompositeMPSTensor}) = [t' for t in ts]
Base.adjoint(ts::Vector{<:AdjointCompositeMPSTensor}) = [t' for t in ts]

mutable struct DenseMPOTensor{S<:Number, R, T<:AbstractTensorMap} <: AbstractMPOTensor
    A::T

    function DenseMPOTensor(t::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(t), rank(t), TM}(t)
    end

    function DenseMPOTensor{S}(t::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, rank(t), TM}(t)
    end

    function DenseMPOTensor{S, R, TM}(t::TM) where {S<:Number, R, TM<:AbstractTensorMap}
        return new{S, R, TM}(t)
    end

    function DenseMPOTensor(fc::Function, codom, dom; type::Type{<:Number} = scalartype(codom))
        A = TensorMap(fc, type, codom, dom)
        return new{scalartype(A), rank(A), typeof(A)}(A)
    end
end


mutable struct AdjointMPOTensor{S<:Number, R, T<:AbstractTensorMap} <: AbstractMPOTensor
    A::T

    function AdjointMPOTensor(t::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(t), rank(t), TM}(t)
    end

    function AdjointMPOTensor(fc::Function, codomain::Union{VectorSpace,ElementarySpace}, domain::Union{VectorSpace,ElementarySpace}; type::Type{<:Number} = scalartype(codomain))
        A = TensorMap(fc, type, codomain, domain)
        return new{scalartype(A), rank(A), typeof(A)}(A)
    end

    function AdjointMPOTensor{S}(t::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, rank(t), TM}(t)
    end

    function AdjointMPOTensor{S, R, TM}(t::TM) where {S<:Number, R, TM<:AbstractTensorMap}
        return new{S, R, TM}(t)
    end
end

Base.adjoint(t::DenseMPOTensor) = AdjointMPOTensor(t.A')
Base.adjoint(t::AdjointMPOTensor) = DenseMPOTensor(t.A')
Base.adjoint(ts::Vector{<:DenseMPOTensor}) = [t' for t in ts]
Base.adjoint(ts::Vector{<:AdjointMPOTensor}) = [t' for t in ts]

mutable struct CompositeMPOTensor{S<:Number, N, R, T<:AbstractTensorMap} <: AbstractMPOTensor
    A::T

    function CompositeMPOTensor(A::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(A), length(codomain(A))-1, rank(A), TM}(A)
    end

    function CompositeMPOTensor{S}(A::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, length(codomain(A))-1, rank(A), TM}(A)
    end

    function CompositeMPOTensor{S, N, R, TM}(A::TM) where {S<:Number, N, R, TM<:AbstractTensorMap}
        return new{S, N, R, TM}(A)
    end

    function CompositeMPOTensor(fc::Function, codom, dom; type::Type{<:Number} = scalartype(codom))
        A = TensorMap(fc, type, codom, dom)
        return new{scalartype(A), length(codomain(A))-1, rank(A), typeof(A)}(A)
    end
end

mutable struct AdjointCompositeMPOTensor{S<:Number, N, R, T<:AbstractTensorMap} <: AbstractMPOTensor
    A::T

    function AdjointCompositeMPOTensor(A::TM) where {TM<:AbstractTensorMap}
        return new{scalartype(A), length(domain(A))-1, rank(A), TM}(A)
    end

    function AdjointCompositeMPOTensor{S}(A::TM) where {S<:Number, TM<:AbstractTensorMap}
        return new{S, length(domain(A))-1, rank(A), TM}(A)
    end

    function AdjointCompositeMPOTensor{S, N, R, TM}(A::TM) where {S<:Number, N, R, TM<:AbstractTensorMap}
        return new{S, N, R, TM}(A)
    end

    function AdjointCompositeMPOTensor(fc::Function, codom, dom; type::Type{<:Number} = scalartype(codom))
        A = TensorMap(fc, type, codom, dom)
        return new{scalartype(A), length(domain(A))-1, rank(A), typeof(A)}(A)
    end
end

Base.adjoint(t::CompositeMPOTensor) = AdjointCompositeMPOTensor(t.A')
Base.adjoint(t::AdjointCompositeMPOTensor) = CompositeMPOTensor(t.A')
Base.adjoint(ts::Vector{<:CompositeMPOTensor}) = [t' for t in ts]
Base.adjoint(ts::Vector{<:AdjointCompositeMPOTensor}) = [t' for t in ts]

mutable struct SparseMPOTensor{N,M} <: AbstractMPOTensor
    m::Matrix{Union{Nothing, AbstractLocalOperator}}

    function SparseMPOTensor(m::Matrix{Union{Nothing,AbstractLocalOperator}})
        return new{size(m)...}(m::Matrix{Union{Nothing,AbstractLocalOperator}})
    end

    function SparseMPOTensor(::Nothing, N::Int64, M::Int64)
        return new{N,M}(Matrix{Union{Nothing,AbstractLocalOperator}}(nothing, N, M))
    end
end


#= ====================== =#
