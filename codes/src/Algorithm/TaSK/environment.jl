
mutable struct TaSKEnvironment{L, T <: AbstractTensorWrapper, T′ <: Union{DenseMPO, SparseMPO}} <: AbstractEnvironment
    TC::Vector{T}
    TL::Vector{T}
    TR::Vector{T}
    EnvL::Vector{<:AbstractLeftEnvironmentTensor}
    EnvR::Vector{<:AbstractRightEnvironmentTensor}
    OrthL::Vector{<:AbstractLeftEnvironmentTensor}
    OrthR::Vector{<:AbstractRightEnvironmentTensor}
    H::T′
    Eg::Float64
    n::Int64
    isdisk::Bool
    function TaSKEnvironment(TC::Vector{T}, TL::Vector{T}, TR::Vector{T}, H::SparseMPO{L}, Eg::Float64 = 0.0, isdisk::Bool = IS_DISK[]) where {T <: Union{MPSTensor,DenseMPOTensor},L}
        @assert L == length(TC) == length(TL) == length(TR)
        return new{L,T,SparseMPO}(
            TC,TL,TR,
            Vector{AbstractLeftEnvironmentTensor}(),Vector{AbstractRightEnvironmentTensor}(),
            Vector{AbstractLeftEnvironmentTensor}(),Vector{AbstractRightEnvironmentTensor}(),
            H,Eg,0,isdisk
        )
    end
    function TaSKEnvironment(env::TaSKEnvironment{L,T,T′}) where {L,T,T′}
        return new{L,T,T′}(deepcopy(env.TC),env.TL,env.TR,env.EnvL,env.EnvR,env.OrthL,env.OrthR,env.H,env.Eg,env.n,env.isdisk)
    end
end

Base.copy(env::TaSKEnvironment) = TaSKEnvironment(env)

function Base.show(io::IO, env::TaSKEnvironment{L,T,T′}) where {L,T,T′}
    print(io, "$(typeof(env))\n")
    println(io, " - ","T : $(length(env.EnvL)) x $(T)")
    println(io, " - ","EnvL: 2 x $(length(env.EnvL)) x $(typeof(env.EnvL[1]))")
    println(io, " - ","EnvR: 2 x $(length(env.EnvR)) x $(typeof(env.EnvR[1]))")
    println(io, " - ","H : $(T′)")
    println(io, " - ","n : $(env.n)")
end

function TaSKEnvironment(A::T, O::SparseMPO{L}, Eg::Float64 = 0.0) where T <: Union{DenseMPS{L}, DenseMPO{L}} where L
    Tt = typeof(A[1])
    TC = Tt[]
    TL = Tt[]
    TR = Tt[]

    (center = A.center[1]) ≠ 1 && canonicalize!(A,1)
    push!(TR,rightorth(A[1])[2])
    for i in 1:L
        canonicalize!(A,i)
        i > 1 && push!(TL,A[i-1])
        i < L && push!(TR,A[i+1])
        push!(TC,A[i])
    end
    push!(TL,leftorth(A[end])[1])
    canonicalize!(A,1)

    env = TaSKEnvironment(TC,TL,TR,O,Eg)
    initialize!(env)
    return env
end
