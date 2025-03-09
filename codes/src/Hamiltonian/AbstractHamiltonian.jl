

mutable struct SparseProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
    EnvL::SparseLeftEnvironmentTensor
    EnvR::SparseRightEnvironmentTensor
    H::Union{Nothing,SparseMPO}

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor,
        EnvR::SparseRightEnvironmentTensor,
        H::SparseMPO) 
        return new{length(H.ts)}(EnvL,EnvR,H)
    end

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor,
        EnvR::SparseRightEnvironmentTensor) 
        return new{0}(EnvL,EnvR,nothing)
    end
end

proj0(EnvL,EnvR) = SparseProjectiveHamiltonian(EnvL,EnvR)

function projleft0(env::Environment{3})
    !issparse(env.layer[2]) && return nothing
    site = env.center[1]
    EnvR = pushleft(map(x -> env.layer[x],eachindex(env.layer))...,env.envs[site+1],site)
    return SparseProjectiveHamiltonian(env.envs[site],EnvR)
end

function projright0(env::Environment{3})
    !issparse(env.layer[2]) && return nothing
    site = env.center[1]
    EnvL = pushright(map(x -> env.layer[x],eachindex(env.layer))...,env.envs[site],site)
    return SparseProjectiveHamiltonian(EnvL,env.envs[site+1])
end

proj1(env::Environment{3},site::Int64) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[site:site+1]...,SparseMPO(env.layer[2].ts[site])) : nothing

function proj2(env::Environment{3},site1::Int64,site2::Int64)
    !issparse(env.layer[2]) && return nothing
    return site1 < site2 ? projright2(env,site1) : projleft2(env,site2)
end
projright2(env::Environment{3},site::Int64) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[[site,site+2]]...,SparseMPO(env.layer[2].ts[site:site+1])) : nothing
projleft2(env::Environment{3},site::Int64) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[[site-1,site+1]]...,SparseMPO(env.layer[2].ts[site-1:site])) : nothing

