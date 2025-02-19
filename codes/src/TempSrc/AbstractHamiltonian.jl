
abstract type AbstractHamiltonian end
abstract type AbstractProjectiveHamiltonian <: AbstractHamiltonian end
mutable struct SparseProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
    EnvL::SparseLeftEnvironmentTensor
    EnvR::SparseRightEnvironmentTensor
    H::SparseMPO

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor,
        EnvR::SparseRightEnvironmentTensor,
        H::SparseMPO) 
        return new{length(H.ts)}(EnvL,EnvR,H)
    end
end

function proj1(env::Environment{3},site::Int64)
    issparse(env.layer[2]) && return SparseProjectiveHamiltonian(env.envs[site:site+1]...,SparseMPO(env.layer[2].ts[site]))
end

function projright2(env::Environment{3},site::Int64)
    issparse(env.layer[2]) && return SparseProjectiveHamiltonian(env.envs[[site,site+2]]...,SparseMPO(env.layer[2].ts[site:site+1]))
end

function projleft2(env::Environment{3},site::Int64)
    issparse(env.layer[2]) && return SparseProjectiveHamiltonian(env.envs[[site-1,site+1]]...,SparseMPO(env.layer[2].ts[site-1:site]))
end

