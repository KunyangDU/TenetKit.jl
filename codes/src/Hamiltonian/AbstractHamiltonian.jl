

mutable struct SparseProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
    EnvL::SparseLeftEnvironmentTensor
    EnvR::SparseRightEnvironmentTensor
    H::Union{Nothing,SparseMPO}
    validinds::Tuple
    E₀::Number

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor{1},
        EnvR::SparseRightEnvironmentTensor{1},
        H::SparseMPO{2},E₀::Number = 0.0) 
        N,M1 = H.D[1]
        M2,R = H.D[2]
        @assert M1 == M2
        @assert EnvL.D[1] == N 
        @assert EnvR.D[1] == R

        viv = []
        for i in 1:N,j in 1:M1, k in 1:R
            isnothing(H.ts[1].m[i,j]) | isnothing(H.ts[2].m[j,k]) && continue
            push!(viv,(i,j,k))
        end

        return new{2}(EnvL,EnvR,H,Tuple(viv),E₀)
    end

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor{1},
        EnvR::SparseRightEnvironmentTensor{1},
        H::SparseMPO{1},E₀::Number = 0.0)  
        N,R = H.D[1]
        @assert EnvL.D[1] == N 
        @assert EnvR.D[1] == R

        viv = []
        for i in 1:N, j in 1:R
            isnothing(H.ts[1].m[i,j]) && continue
            push!(viv,(i,j))
        end

        return new{1}(EnvL,EnvR,H,Tuple(viv),E₀)
    end

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor{1},
        EnvR::SparseRightEnvironmentTensor{1},E₀::Number = 0.0)  
        N = EnvL.D[1]
        M = EnvR.D[1]
        @assert M == N 
        
        return new{0}(EnvL,EnvR,nothing,Tuple(1:N),E₀)
    end
end

proj0(EnvL,EnvR;E₀::Number = 0.0) = SparseProjectiveHamiltonian(EnvL,EnvR,E₀)

function projleft0(env::Environment{3};E₀::Number = 0.0)
    site = env.center[1]
    EnvR = pushleft(map(x -> env.layer[x],eachindex(env.layer))...,env.envs[site+1],site)
    return issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[site],EnvR,E₀) : DenseProjectiveHamiltonian(env.envs[site],EnvR,E₀)
end

function projright0(env::Environment{3};E₀::Number = 0.0)
    site = env.center[1]
    EnvL = pushright(map(x -> env.layer[x],eachindex(env.layer))...,env.envs[site],site)
    return issparse(env.layer[2]) ? SparseProjectiveHamiltonian(EnvL,env.envs[site+1],E₀) : DenseProjectiveHamiltonian(EnvL,env.envs[site+1],E₀)
end

proj1(env::Environment{3},site::Int64;E₀::Number = 0.0) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[site:site+1]...,SparseMPO(env.layer[2].ts[site]),E₀) : DenseProjectiveHamiltonian(env.envs[site:site+1]...,[env.layer[2].ts[site],],E₀)

function proj2(env::Environment{3},site1::Int64,site2::Int64;E₀::Number = 0.0)
    # !issparse(env.layer[2]) && return nothing
    return site1 < site2 ? projright2(env,site1,E₀) : projleft2(env,site2,E₀)
end
projright2(env::Environment{3},site::Int64,E₀::Number = 0.0) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[[site,site+2]]...,SparseMPO(env.layer[2].ts[site:site+1]),E₀) : DenseProjectiveHamiltonian(env.envs[[site,site+2]]...,env.layer[2].ts[site:site+1],E₀)
projleft2(env::Environment{3},site::Int64,E₀::Number = 0.0) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[[site-1,site+1]]...,SparseMPO(env.layer[2].ts[site-1:site]),E₀) : DenseProjectiveHamiltonian(env.envs[[site-1,site+1]]...,env.layer[2].ts[site-1:site],E₀)
proj2(EnvL::SparseLeftEnvironmentTensor,hl::SparseMPOTensor,hr::SparseMPOTensor,EnvR::SparseRightEnvironmentTensor;E₀::Number = 0.) = SparseProjectiveHamiltonian(EnvL,EnvR,SparseMPO([hl,hr]),E₀)
proj2(EnvL::DenseLeftEnvironmentTensor,hl::DenseMPOTensor,hr::DenseMPOTensor,EnvR::DenseRightEnvironmentTensor;E₀::Number = 0.) = DenseProjectiveHamiltonian(EnvL,EnvR,[hl,hr],E₀)

function proj2(env::Environment{2},site1::Int64,site2::Int64;E₀::Number = 0.0)
    !issparse(env.layer[1]) && return nothing
    return site1 < site2 ? projright2(env,site1,E₀) : projleft2(env,site2,E₀)
end
projright2(env::Environment{2},site::Int64,E₀::Number = 0.0) = issparse(env.layer[1]) ? SparseProjectiveHamiltonian(env.envs[[site,site+2]]...,SparseMPO(env.layer[1].ts[site:site+1]),E₀) : nothing
projleft2(env::Environment{2},site::Int64,E₀::Number = 0.0) = issparse(env.layer[1]) ? SparseProjectiveHamiltonian(env.envs[[site-1,site+1]]...,SparseMPO(env.layer[1].ts[site-1:site]),E₀) : nothing

mutable struct DenseProjectiveHamiltonian{N,L} <: AbstractProjectiveHamiltonian
    EnvL::DenseLeftEnvironmentTensor
    EnvR::DenseRightEnvironmentTensor
    H::Union{Nothing,Array}
    E₀::Number

    function DenseProjectiveHamiltonian(EnvL::DenseLeftEnvironmentTensor,
        EnvR::DenseRightEnvironmentTensor,
        H::Array,E₀::Number = 0.0) 
        return new{3,length(H)}(EnvL,EnvR,H,E₀)
    end

    function DenseProjectiveHamiltonian(EnvL::DenseLeftEnvironmentTensor,
        EnvR::DenseRightEnvironmentTensor, E₀::Number = 0.0) 
        return new{3,0}(EnvL,EnvR,nothing,E₀)
    end

    function DenseProjectiveHamiltonian{N,L}(EnvL::DenseLeftEnvironmentTensor,
        EnvR::DenseRightEnvironmentTensor,E₀::Number = 0.0) where {N,L}
        return new{N,L}(EnvL,EnvR,nothing,E₀)
    end
end

proj1(env::Environment{2},site::Int64) = DenseProjectiveHamiltonian{2,1}(env.envs[site:site+1]...)

