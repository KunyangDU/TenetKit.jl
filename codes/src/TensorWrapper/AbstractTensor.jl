
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
        return new{numind(ts)}(ts)
    end

    function MPSTensor{r}(ts::AbstractTensorMap) where r
        return new{r}(ts)
    end

    function MPSTensor(fc::Function,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = fc(Float64,codomain,domain)
        return new{numind(A)}(A)
    end

    function MPSTensor(data::AbstractMatrix,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(data[:],codomain,domain)
        return new{numind(A)}(A)
    end

    function MPSTensor(data::AbstractVector,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = TensorMap(data,codomain,domain)
        return new{numind(A)}(A)
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
        return new{numind(ts)}(ts)
    end

    function AdjointMPSTensor(fc::Function,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = fc(codomain,domain)
        return new{numind(A)}(A)
    end

end

Base.adjoint(t::MPSTensor) = AdjointMPSTensor(copy(t.A'))
Base.adjoint(ts::Vector{MPSTensor}) = convert(Vector{AdjointMPSTensor},[AdjointMPSTensor(copy(t.A')) for t in ts])
Base.adjoint(ts::AbstractVector{MPSTensor}) = convert(Vector{AdjointMPSTensor},[AdjointMPSTensor(copy(t.A')) for t in ts])
Base.adjoint(t::AdjointMPSTensor) = MPSTensor(copy(t.A'))
Base.adjoint(ts::Vector{AdjointMPSTensor}) = convert(Vector{MPSTensor},[MPSTensor(copy(t.A')) for t in ts])
Base.adjoint(ts::AbstractVector{AdjointMPSTensor}) = convert(Vector{MPSTensor},[MPSTensor(copy(t.A')) for t in ts])

"""
todo {}
    1' - A - R
         | \
         2' 3'...(R-1)'
"""
mutable struct CompositeMPSTensor{N, R} <: AbstractMPSTensor
    A::AbstractTensorMap

    function CompositeMPSTensor(A::AbstractTensorMap)
        return new{numout(A)-1, numind(A)}(A)
    end
    function CompositeMPSTensor{n,r}(A::AbstractTensorMap) where {n,r}
        return new{n,r}(A)
    end

    function CompositeMPSTensor(fc::Function, codom, dom)
        A = fc(codom,dom)
        return new{numout(A)-1, numind(A)}(A)
    end
end

mutable struct AdjointCompositeMPSTensor{N, R} <: AbstractMPSTensor
    A::AbstractTensorMap

    function AdjointCompositeMPSTensor(A::AbstractTensorMap)
        return new{numin(A)-1, numind(A)}(A)
    end

    function AdjointCompositeMPSTensor{n,r}(A::AbstractTensorMap) where {n,r}
        return new{n, r}(A)
    end

    function AdjointCompositeMPSTensor(fc::Function,codom,dom)
        A = fc(codom,dom)
        return new{numin(A)-1, numind(A)}(A)
    end
end

Base.adjoint(t::CompositeMPSTensor) = AdjointCompositeMPSTensor(copy(t.A'))
Base.adjoint(t::AdjointCompositeMPSTensor) = CompositeMPSTensor(copy(t.A'))
Base.adjoint(ts::Vector{CompositeMPSTensor}) = convert(Vector{AdjointCompositeMPSTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{CompositeMPSTensor}) = convert(Vector{AdjointCompositeMPSTensor},[t' for t in ts])
Base.adjoint(ts::Vector{AdjointCompositeMPSTensor}) = convert(Vector{CompositeMPSTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{AdjointCompositeMPSTensor}) = convert(Vector{CompositeMPSTensor},[t' for t in ts])

mutable struct DenseMPOTensor{R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function DenseMPOTensor(t::AbstractTensorMap)
        return new{numind(t)}(t)
    end

    function DenseMPOTensor{r}(t::AbstractTensorMap) where r
        return new{r}(t)
    end
    function DenseMPOTensor(fc::Function,codom,dom)
        A = fc(codom,dom)
        return new{numind(A)}(A)
    end
end


mutable struct AdjointMPOTensor{R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function AdjointMPOTensor(t::AbstractTensorMap)
        return new{numind(t)}(t)
    end

    function AdjointMPOTensor(fc::Function,codomain::Union{VectorSpace,ElementarySpace},domain::Union{VectorSpace,ElementarySpace})
        A = fc(codomain,domain)
        return new{numind(A)}(A)
    end

    function AdjointMPOTensor{r}(t::AbstractTensorMap) where r
        return new{r}(t)
    end
end

Base.adjoint(t::DenseMPOTensor) = AdjointMPOTensor(copy(t.A'))
Base.adjoint(t::AdjointMPOTensor) = DenseMPOTensor(copy(t.A'))
Base.adjoint(ts::Vector{DenseMPOTensor}) = convert(Vector{AdjointMPOTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{DenseMPOTensor}) = convert(Vector{AdjointMPOTensor},[t' for t in ts])
Base.adjoint(ts::Vector{AdjointMPOTensor}) = convert(Vector{DenseMPOTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{AdjointMPOTensor}) = convert(Vector{DenseMPOTensor},[t' for t in ts])

mutable struct CompositeMPOTensor{N, R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function CompositeMPOTensor(A::AbstractTensorMap)
        return new{numout(A)-1, numind(A)}(A)
    end

    function CompositeMPOTensor{n,r}(A::AbstractTensorMap) where {n,r}
        return new{n,r}(A)
    end

    function CompositeMPOTensor(fc::Function, codom, dom)
        A = fc(codom,dom)
        return new{numout(A)-1, numind(A)}(A)
    end
end

mutable struct AdjointCompositeMPOTensor{N, R} <: AbstractMPOTensor
    A::AbstractTensorMap

    function AdjointCompositeMPOTensor(A::AbstractTensorMap)
        return new{numin(A)-1, numind(A)}(A)
    end

    function AdjointCompositeMPOTensor{n,r}(A::AbstractTensorMap) where {n,r}
        return new{n,r}(A)
    end

    function AdjointCompositeMPOTensor(fc::Function,codom,dom)
        A = fc(codom,dom)
        return new{numin(A)-1, numind(A)}(A)
    end
end

Base.adjoint(t::CompositeMPOTensor) = AdjointCompositeMPOTensor(copy(t.A'))
Base.adjoint(t::AdjointCompositeMPOTensor) = CompositeMPOTensor(copy(t.A'))
Base.adjoint(ts::Vector{CompositeMPOTensor}) = convert(Vector{AdjointCompositeMPOTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{CompositeMPOTensor}) = convert(Vector{AdjointCompositeMPOTensor},[t' for t in ts])
Base.adjoint(ts::Vector{AdjointCompositeMPOTensor}) = convert(Vector{CompositeMPOTensor},[t' for t in ts])
Base.adjoint(ts::AbstractVector{AdjointCompositeMPOTensor}) = convert(Vector{CompositeMPOTensor},[t' for t in ts])

composite(A::MPSTensor{3}, B::MPSTensor{3}) = CompositeMPSTensor(@tensor tmp[-1 -2 -3; -4] ≔ A.A[-1,-2,1]*B.A[1,-3,-4])
composite(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}) = CompositeMPOTensor(@tensor tmp[-1 -2 -3;-4 -5 -6] ≔ A.A[-2,-3,1,-6] * B.A[-1,1,-4,-5])
composite(A::T, B::T) where T <: Union{AdjointMPSTensor{3},AdjointMPOTensor{4}} = composite(A',B')'


mutable struct SparseMPOTensor{DL,D,DR,T} <: AbstractMPOTensor
    A::Vector{AbstractLocalOperator}
    left::LayerMap{1,DL,D,T}    # 左 bond 节点 ↔ 本层算符
    right::LayerMap{1,D,DR,T}   # 本层算符 ↔ 右 bond 节点
    SparseMPOTensor(A::Vector, left::LayerMap{1,DL,D,T}, right::LayerMap{1,D,DR,T}) where {DL,D,DR,T} = new{DL,D,DR,T}(A, left, right)
end

Base.length(::SparseMPOTensor{DL,D,DR,T}) where {DL,D,DR,T} = D
Base.eachindex(h::SparseMPOTensor) = Base.OneTo(length(h))
Base.getindex(obj::SparseMPOTensor, i::Int64) = obj.A[i]
Base.adjoint(A::SparseMPOTensor) = return SparseMPOTensor(adjoint.(A.A), A.left, A.right)
