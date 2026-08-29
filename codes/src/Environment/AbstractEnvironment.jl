
"""
Monolayer Environment, i.e., only one layer MPO is considered.
"""
mutable struct Environment{N,L} <: AbstractEnvironment
    layer::Vector
    envs::Union{Nothing,AbstractArray{AbstractEnvironmentTensor}}
    center::Vector{Int64}
    L::Int64
    isdisk::Bool

    function Environment(layer::Vector,
        envs::AbstractArray{AbstractEnvironmentTensor},
        center::Union{Nothing,Vector{Int64}},
        L::Union{Nothing,Int64}; isdisk::Bool=IS_DISK[])
        if isdisk && envs isa Vector
            envs = _disk(envs)
        end
        return new{length(layer),length(layer[1])}(layer,envs,center,L,isdisk)
    end

    function Environment(layer::Vector; isdisk::Bool=IS_DISK[])
        L = length(layer[1])
        return new{length(layer),length(layer[1])}(layer,nothing,[1,L],L,isdisk)
    end

    # 变参快捷构造：Environment(A,B,C) ≡ Environment([A,B,C])（至少一层）
    function Environment(layer, layers...; isdisk::Bool=IS_DISK[])
        return Environment([layer, layers...]; isdisk=isdisk)
    end
end

function cleanup!(env::Environment)
    if env.isdisk && env.envs isa SerializedElementArrays.SerializedElementArray
        dir = SerializedElementArrays.pathname(env.envs)
        ispath(dir) && rm(dir; recursive=true, force=true)
    end
    return env
end

mutable struct CBEenvironment <: AbstractEnvironment
    tL₀::AbstractTensorWrapper
    tR₀::AbstractTensorWrapper
    tL::Union{AbstractTensorWrapper,SparseMPOTensor,Nothing}
    tR::Union{AbstractTensorWrapper,SparseMPOTensor,Nothing}
    D_i::Int64
    D_f::Int64
    Λ::Union{AbstractTensorWrapper,Nothing}
    Lorth::Union{SparseLeftEnvironmentTensor,LeftCompositeEnvironmentTensor,LeftEnvironmentTensor,DenseLeftEnvironmentTensor,Nothing}
    Rorth::Union{SparseRightEnvironmentTensor,RightCompositeEnvironmentTensor,RightEnvironmentTensor,DenseRightEnvironmentTensor,Nothing}
    lm::Union{LayerMap,Nothing}
end

