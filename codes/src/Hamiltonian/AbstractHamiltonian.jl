

mutable struct SparseProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
    EnvL::SparseLeftEnvironmentTensor
    EnvR::SparseRightEnvironmentTensor
    H::Union{Nothing,SparseMPO}
    validinds::Tuple
    E₀::Number

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor{1},
        EnvR::SparseRightEnvironmentTensor{1},
        H::SparseMPO{2},E₀::Number = 0.0)
        DL1,D1,DR1 = H.D[1]
        DL2,D2,DR2 = H.D[2]
        @assert EnvL.D[1] == DL1
        @assert EnvR.D[1] == DR2
        return new{2}(EnvL,EnvR,H,Tuple(_validind(H[1], H[2])),E₀)
    end

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor{1},
        EnvR::SparseRightEnvironmentTensor{1},
        H::SparseMPO{1},E₀::Number = 0.0)
        DL,D,DR = H.D[1]
        @assert EnvL.D[1] == DL EnvL.D[1],DL
        @assert EnvR.D[1] == DR EnvR.D[1],DR

        return new{1}(EnvL,EnvR,H,Tuple(_validind(H[1])),E₀)
    end

    function SparseProjectiveHamiltonian(EnvL::SparseLeftEnvironmentTensor{1},
        EnvR::SparseRightEnvironmentTensor{1},
        lm::LayerMap{N,D₁,D₂}, E₀::Number = 0.0) where {N,D₁,D₂}
        @assert EnvL.D[1] == D₁
        @assert EnvR.D[1] == D₂
        return new{0}(EnvL,EnvR,nothing,Tuple(_validind0(lm)),E₀)
    end
end

proj0(EnvL,EnvR;E₀::Number = 0.0) = SparseProjectiveHamiltonian(EnvL,EnvR,E₀)

function projleft0(env::Environment{3};E₀::Number = 0.0)
    site = env.center[1]
    EnvR = pushleft(map(x -> env.layer[x],eachindex(env.layer))...,env.envs[site+1],site)
    if issparse(env.layer[2])
        lm = env.layer[2][site].left
        return SparseProjectiveHamiltonian(env.envs[site], EnvR, lm, E₀)
    else
        return DenseProjectiveHamiltonian(env.envs[site], EnvR, E₀)
    end
end

function projright0(env::Environment{3};E₀::Number = 0.0)
    site = env.center[1]
    EnvL = pushright(map(x -> env.layer[x],eachindex(env.layer))...,env.envs[site],site)
    if issparse(env.layer[2])
        lm = env.layer[2][site].right
        return SparseProjectiveHamiltonian(EnvL, env.envs[site+1], lm, E₀)
    else
        return DenseProjectiveHamiltonian(EnvL, env.envs[site+1], E₀)
    end
end

proj1(env::Environment{3},site::Int64;E₀::Number = 0.0) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[site],env.envs[site+1],SparseMPO(env.layer[2][site]),E₀) : DenseProjectiveHamiltonian(env.envs[site],env.envs[site+1],[env.layer[2][site],],E₀)

function proj2(env::Environment{3},site1::Int64,site2::Int64;E₀::Number = 0.0)
    # !issparse(env.layer[2]) && return nothing
    return site1 < site2 ? projright2(env,site1,E₀) : projleft2(env,site2,E₀)
end
projright2(env::Environment{3},site::Int64,E₀::Number = 0.0) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[site],env.envs[site+2],SparseMPO(env.layer[2][site:site+1]),E₀) : DenseProjectiveHamiltonian(env.envs[site],env.envs[site+2],env.layer[2][site:site+1],E₀)
projleft2(env::Environment{3},site::Int64,E₀::Number = 0.0) = issparse(env.layer[2]) ? SparseProjectiveHamiltonian(env.envs[site-1],env.envs[site+1],SparseMPO(env.layer[2][site-1:site]),E₀) : DenseProjectiveHamiltonian(env.envs[site-1],env.envs[site+1],env.layer[2][site-1:site],E₀)
proj2(EnvL::SparseLeftEnvironmentTensor,hl::SparseMPOTensor,hr::SparseMPOTensor,EnvR::SparseRightEnvironmentTensor;E₀::Number = 0.) = SparseProjectiveHamiltonian(EnvL,EnvR,SparseMPO([hl,hr]),E₀)
proj2(EnvL::DenseLeftEnvironmentTensor,hl::DenseMPOTensor,hr::DenseMPOTensor,EnvR::DenseRightEnvironmentTensor;E₀::Number = 0.) = DenseProjectiveHamiltonian(EnvL,EnvR,[hl,hr],E₀)
proj2(EnvL::DenseLeftEnvironmentTensor{2}, ::Nothing, ::Nothing, EnvR::DenseRightEnvironmentTensor{2};E₀::Number = 0.) = DenseProjectiveHamiltonian{2,2}(EnvL,EnvR,E₀)

function proj2(env::Environment{2},site1::Int64,site2::Int64;E₀::Number = 0.0)
    !issparse(env.layer[1]) && return nothing
    return site1 < site2 ? projright2(env,site1,E₀) : projleft2(env,site2,E₀)
end
projright2(env::Environment{2},site::Int64,E₀::Number = 0.0) = issparse(env.layer[1]) ? SparseProjectiveHamiltonian(env.envs[site],env.envs[site+2],SparseMPO(env.layer[1][site:site+1]),E₀) : nothing
projleft2(env::Environment{2},site::Int64,E₀::Number = 0.0) = issparse(env.layer[1]) ? SparseProjectiveHamiltonian(env.envs[site-1],env.envs[site+1],SparseMPO(env.layer[1][site-1:site]),E₀) : nothing

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

proj1(env::Environment{2},site::Int64) = DenseProjectiveHamiltonian{2,1}(env.envs[site],env.envs[site+1])

proj1(EnvL::SparseLeftEnvironmentTensor, H::SparseMPOTensor, EnvR::SparseRightEnvironmentTensor, E₀::Float64 = 0.0) = SparseProjectiveHamiltonian(EnvL,EnvR,SparseMPO(H),E₀)
