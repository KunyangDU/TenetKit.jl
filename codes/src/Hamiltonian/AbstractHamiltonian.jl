

mutable struct SparseProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
    EnvL::SparseLeftEnvironmentTensor
    EnvR::SparseRightEnvironmentTensor
    H::Union{Nothing,SparseMPO}
    validinds::Tuple

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor,
        EnvR::SparseRightEnvironmentTensor,
        H::SparseMPO{2}) 
        N,M1 = H.D[1]
        M2,R = H.D[2]
        @assert M1 == M2
        @assert EnvL.D == N 
        @assert EnvR.D == R

        viv = []
        for i in 1:N,j in 1:M1, k in 1:R
            isnothing(H.ts[1].m[i,j]) | isnothing(H.ts[2].m[j,k]) && continue
            push!(viv,(i,j,k))
        end

        return new{2}(EnvL,EnvR,H,Tuple(viv))
    end

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor,
        EnvR::SparseRightEnvironmentTensor,
        H::SparseMPO{1}) 
        N,R = H.D[1]
        @assert EnvL.D == N 
        @assert EnvR.D == R

        viv = []
        for i in 1:N, j in 1:R
            isnothing(H.ts[1].m[i,j]) && continue
            push!(viv,(i,j))
        end

        return new{1}(EnvL,EnvR,H,Tuple(viv))
    end

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor,
        EnvR::SparseRightEnvironmentTensor) 
        N = EnvL.D
        M = EnvR.D
        @assert M == N 
        
        return new{0}(EnvL,EnvR,nothing,Tuple(1:N))
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
proj2(EnvL::SparseLeftEnvironmentTensor,hl::SparseMPOTensor,hr::SparseMPOTensor,EnvR::SparseRightEnvironmentTensor) = SparseProjectiveHamiltonian(EnvL,EnvR,SparseMPO([hl,hr]))

mutable struct DenseProjectiveHamiltonian{N,L} <: AbstractProjectiveHamiltonian
    EnvL::DenseLeftEnvironmentTensor
    EnvR::DenseRightEnvironmentTensor
    H::Union{Nothing,Array}

    function DenseProjectiveHamiltonian(EnvL::DenseLeftEnvironmentTensor,
        EnvR::DenseRightEnvironmentTensor,
        H::Array) 
        return new{3,length(H.ts)}(EnvL,EnvR,H)
    end

    function DenseProjectiveHamiltonian{N,L}(EnvL::DenseLeftEnvironmentTensor,
        EnvR::DenseRightEnvironmentTensor) where {N,L}
        return new{N,L}(EnvL,EnvR,nothing)
    end
end

proj1(env::Environment{2},site::Int64) = DenseProjectiveHamiltonian{2,1}(env.envs[site:site+1]...)

