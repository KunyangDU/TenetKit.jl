function pushleft!(env::Environment{R}) where R
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[2]] = pushleft(env.layer..., env.envs[env.center[2] + 1], env.center[2])

    env.center[2] -= 1
    ( env.center[1] > env.center[2] ) && ( env.center[1] -= 1 )
end

function pushleft(A::AbstractMPS, mpo::SparseMPO, B::AbstractMPS, EnvR::SparseRightEnvironmentTensor, site::Int64)
    @assert mpo.D[site][2] == EnvR.D
    tmpEnvR = Vector{Any}(nothing,mpo.D[site][1])
    for i in eachindex(tmpEnvR), j in 1:EnvR.D
        isnothing(mpo.ts[site].m[i,j]) && continue
        if isnothing(tmpEnvR[i])
            tmpEnvR[i] = contract(A.ts[site], mpo.ts[site].m[i,j], B.ts[site], EnvR.A[j])
        else 
            tmpEnvR[i] += contract(A.ts[site], mpo.ts[site].m[i,j], B.ts[site], EnvR.A[j])
        end
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor},tmpEnvR))
end


function pushright!(env::Environment{R}) where R
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[1] + 1] = pushright(env.layer..., env.envs[env.center[1]], env.center[1])

    env.center[1] += 1
    ( env.center[1] > env.center[2] ) && ( env.center[2] += 1 )
end

function pushright(A::AbstractMPS, mpo::SparseMPO, B::AbstractMPS, EnvL::SparseLeftEnvironmentTensor, site::Int64)
    @assert mpo.D[site][1] == EnvL.D
    tmpEnvL = Vector{Any}(nothing,mpo.D[site][2])
    for i in eachindex(tmpEnvL), j in 1:EnvL.D
        isnothing(mpo.ts[site].m[j,i]) && continue
        if isnothing(tmpEnvL[i])
            tmpEnvL[i] = contract(A.ts[site], mpo.ts[site].m[j,i], B.ts[site],EnvL.A[j])
        else 
            tmpEnvL[i] += contract(A.ts[site], mpo.ts[site].m[j,i], B.ts[site],EnvL.A[j])
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor},tmpEnvL))
end

function pushright!(env::Environment{N}, tl::DenseMPOTensor{4}, tr::DenseMPOTensor{4}) where N
    @assert (site = env.center[1] ) == env.center[2]
    env.layer[end].ts[site:site+1] = map(adjoint,[tl,tr])
    env.layer[end].center = env.center
    map(v -> canonicalize!(env.layer[v],site + 1),1:N-1)
    pushright!(env)
end

function pushleft!(env::Environment{N}, tl::DenseMPOTensor{4}, tr::DenseMPOTensor{4}) where N
    @assert (site = env.center[1] ) == env.center[2]
    env.layer[end].ts[site-1:site] = map(adjoint,[tl,tr])
    env.layer[end].center = env.center
    map(v -> canonicalize!(env.layer[v],site - 1),1:N-1)
    pushleft!(env)
end

function pushleft(A::DenseMPO, B::AdjointMPO, EnvR::DenseRightEnvironmentTensor{2}, site::Int64)
    return DenseRightEnvironmentTensor(contract(map(x -> x.ts[site],(A,B))..., EnvR.A))
end

function pushleft(A::DenseMPO, B::DenseMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64)
    return DenseRightEnvironmentTensor(contract(map(x -> x.ts[site],(A,B,C))..., EnvR.A))
end

function pushright(A::DenseMPO, B::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{2}, site::Int64)
    return DenseLeftEnvironmentTensor(contract(map(x -> x.ts[site],(A,B))..., EnvL.A))
end

function pushright(A::DenseMPO, B::DenseMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64)
    return DenseLeftEnvironmentTensor(contract(map(x -> x.ts[site],(A,B,C))..., EnvL.A))
end

function pushleft(A::DenseMPO, B::SparseMPO, C::AdjointMPO, EnvR::SparseRightEnvironmentTensor, site::Int64)
    return SparseRightEnvironmentTensor(contract(map(x -> x.ts[site],(A,B,C))..., EnvR))
end

function pushright(A::DenseMPO, B::SparseMPO, C::AdjointMPO, EnvL::SparseLeftEnvironmentTensor, site::Int64)
    return SparseLeftEnvironmentTensor(contract(map(x -> x.ts[site],(A,B,C))..., EnvL))
end



