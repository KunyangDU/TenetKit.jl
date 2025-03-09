


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

"""

"""
mutable struct LeftCompositeEnvironmentTensor{N,R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LeftCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{length(codomain(t)),rank(t)}(t)
    end

    function LeftCompositeEnvironmentTensor{n,r}(t::AbstractTensorMap) where {n,r}
        return new{length(codomain(t)),rank(t)}(t)
    end
end

mutable struct RightCompositeEnvironmentTensor{N,R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function RightCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{length(domain(t)),rank(t)}(t)
    end

    function RightCompositeEnvironmentTensor{n,r}(t::AbstractTensorMap) where {n,r}
        return new{length(domain(t)),rank(t)}(t)
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



mutable struct SparseLeftEnvironmentTensor <: AbstractLeftEnvironmentTensor
    A::Union{Vector{LeftEnvironmentTensor},Vector{LeftCompositeEnvironmentTensor}}
    D::Int64

    function SparseLeftEnvironmentTensor(t::Vector{LeftEnvironmentTensor},D::Int64)
        return new(t,D)
    end

    function SparseLeftEnvironmentTensor(t::Union{Vector{LeftEnvironmentTensor},Vector{LeftCompositeEnvironmentTensor}})
        return new(t,length(t))
    end

    function SparseLeftEnvironmentTensor(t::LeftEnvironmentTensor)
        return new(convert(Vector{LeftEnvironmentTensor},[t]),1)
    end

    function SparseLeftEnvironmentTensor(t::AbstractTensorMap)
        return new(convert(Vector{LeftEnvironmentTensor},[LeftEnvironmentTensor(t)]),1)
    end

    function SparseLeftEnvironmentTensor(t::Vector{AbstractTensorMap})
        return new(convert(Vector{LeftEnvironmentTensor},[LeftEnvironmentTensor(ti) for ti in t]),length(t))
    end
end

mutable struct SparseRightEnvironmentTensor <: AbstractRightEnvironmentTensor
    A::Union{Vector{RightEnvironmentTensor},Vector{RightCompositeEnvironmentTensor}}
    D::Int64

    function SparseRightEnvironmentTensor(t::Vector{RightEnvironmentTensor},D::Int64)
        return new(t,D)
    end

    function SparseRightEnvironmentTensor(t::Union{Vector{RightEnvironmentTensor},Vector{RightCompositeEnvironmentTensor}})
        return new(t,length(t))
    end

    function SparseRightEnvironmentTensor(t::RightEnvironmentTensor)
        return new(convert(Vector{RightEnvironmentTensor},[t]),1)
    end

    function SparseRightEnvironmentTensor(t::AbstractTensorMap)
        return new(convert(Vector{RightEnvironmentTensor},[RightEnvironmentTensor(t)]),1)
    end

    function SparseRightEnvironmentTensor(t::Vector{AbstractTensorMap})
        return new(convert(Vector{RightEnvironmentTensor},[RightEnvironmentTensor(ti) for ti in t]),length(t))
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

"""
Monolayer Environment, i.e., only one layer MPO is considered.
"""
mutable struct Environment{N} <: AbstractEnvironment
    layer::Vector
    envs::Union{Nothing,Vector{AbstractEnvironmentTensor}}
    center::Vector{Int64}
    L::Int64

    function Environment(layer::Vector,
        envs::Vector{AbstractEnvironmentTensor},
        center::Union{Nothing,Vector{Int64}},
        L::Union{Nothing,Int64})
        return new{length(layer)}(layer,envs,center,L)
    end

    function Environment(layer::Vector)
        L = length(layer[1])
        return new{length(layer)}(layer,nothing,[1,L],L)
    end

end

