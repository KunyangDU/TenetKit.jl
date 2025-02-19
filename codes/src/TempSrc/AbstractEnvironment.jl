
abstract type AbstractEnvironmentTensor end
abstract type AbstractLeftEnvironmentTensor <: AbstractEnvironmentTensor end
abstract type AbstractRightEnvironmentTensor <: AbstractEnvironmentTensor end

mutable struct LocalEnvironmentTensor{R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LocalEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
end

mutable struct RightEnvironmentTensor{R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function RightEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
end



mutable struct LeftEnvironmentTensor{R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LeftEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(t)
    end
end

function Base.:+(A::LeftEnvironmentTensor,
    B::LeftEnvironmentTensor)
    return LeftEnvironmentTensor(A.A + B.A)
end

function Base.:+(A::RightEnvironmentTensor,
    B::RightEnvironmentTensor)
    return RightEnvironmentTensor(A.A + B.A)
end
"""

"""
mutable struct LeftCompositeEnvironmentTensor{N,R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function LeftCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{length(codomain(t)),rank(t)}(t)
    end
end

mutable struct RightCompositeEnvironmentTensor{N,R} <: AbstractEnvironmentTensor
    A::AbstractTensorMap

    function RightCompositeEnvironmentTensor(t::AbstractTensorMap)
        return new{length(domain(t)),rank(t)}(t)
    end

#=     function RightCompositeEnvironmentTensor(t::AbstractTensorMap,order::Int64)
        return new{order,rank(t)}(t)
    end =#
end

function Base.:+(A::LeftCompositeEnvironmentTensor,
    B::LeftCompositeEnvironmentTensor)
    return LeftCompositeEnvironmentTensor(A.A + B.A)
end

function Base.:+(A::RightCompositeEnvironmentTensor,
    B::RightCompositeEnvironmentTensor)
    return RightCompositeEnvironmentTensor(A.A + B.A)
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

    function DenseLeftEnvironmentTensor(t::LeftEnvironmentTensor)
        return new{rank(t.A)}(t)
    end
end

mutable struct DenseRightEnvironmentTensor{R} <: AbstractLeftEnvironmentTensor
    A::RightEnvironmentTensor

    function DenseRightEnvironmentTensor(t::AbstractTensorMap)
        return new{rank(t)}(RightEnvironmentTensor(t))
    end

    function DenseRightEnvironmentTensor(t::RightEnvironmentTensor)
        return new{rank(t.A)}(t)
    end
end

abstract type AbstractEnvironment end
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

function initialize!(env::Environment)
    env.envs = Vector{AbstractEnvironmentTensor}(undef, env.L + 1)
    setdefault!(env)
    canonicalize!(env,1)
end

function canonicalize!(env::Environment,sl::Int64,sr::Int64)
    @assert 1 ≤ sl ≤ sr ≤ env.L + 1

    for _ in env.center[1]:sl-1
        pushright!(env)
    end

    for _ in env.center[2]:-1:sr+1
        pushleft!(env)
    end

end

function canonicalize!(env::Environment,si::Int64)
    @assert 1 ≤ si ≤ env.L + 1
    canonicalize!(env,si,si)
end


function setdefault!(env::Environment{3})
    if issparse(env.layer[2])
        env.envs[1] = SparseLeftEnvironmentTensor(isometry(reverse(map(x -> getAuxSpace(env.layer[x].ts[1])[1],[1,3]))...))
        env.envs[end] = SparseRightEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[end])[2],[1,3])...))
    else
        AuxSpaces = reverse(map(x -> getAuxSpace(env.layer[x].ts[1])[1],1:3))
        env.envs[1] = DenseLeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2] ⊗ AuxSpaces[3]))
        AuxSpaces = map(x -> getAuxSpace(env.layer[x].ts[end])[2],1:3)
        env.envs[end] = DenseRightEnvironmentTensor(isometry(AuxSpaces[1] ⊗ AuxSpaces[2], AuxSpaces[3]))
    end
end

function setdefault!(env::Environment{2})
    if !issparse(env.layer[1]) && !issparse(env.layer[2])
        env.envs[1] = DenseLeftEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[1])[1],1:2)...))
        env.envs[end] = DenseRightEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[end])[2],1:2)...))
    elseif issparse(env.layer[1]) && !issparse(env.layer[2])
        env.envs[1] = SparseLeftEnvironmentTensor(isometry((getAuxSpace(env.layer[2].ts[1])[1] |> y -> (trivial(y),y))...))
        env.envs[end] = SparseLeftEnvironmentTensor(isometry((getAuxSpace(env.layer[2].ts[end])[2] |> y -> (y,trivial(y)))...))
        #env.envs[end] = SparseRightEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[end])[2],[2,])...))
    end
end
