function pushleft!(env::Environment{R}) where R
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[2]] = pushleft(env.layer..., env.envs[env.center[2] + 1], env.center[2])

    env.center[2] -= 1
    ( env.center[1] > env.center[2] ) && ( env.center[1] -= 1 )
end

function pushright!(env::Environment{R}) where R
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[1] + 1] = pushright(env.layer..., env.envs[env.center[1]], env.center[1])

    env.center[1] += 1
    ( env.center[1] > env.center[2] ) && ( env.center[2] += 1 )
end

pushleft(A::DenseMPS, mpo::SparseMPO, B::T, EnvR::SparseRightEnvironmentTensor{1}, i::Int64) where T <: Union{AdjointMPS,RefMPS} = pushleft(A[i],mpo[i],B[i],EnvR)
pushright(A::DenseMPS, mpo::SparseMPO, B::T, EnvL::SparseLeftEnvironmentTensor{1}, i::Int64) where T <: Union{AdjointMPS,RefMPS} = pushright(A[i],mpo[i],B[i],EnvL)

pushleft(A::DenseMPO, B::SparseMPO, C::T, EnvR::SparseRightEnvironmentTensor, i::Int64) where T <: Union{AdjointMPO,RefMPO} = pushleft(A[i],B[i],C[i],EnvR)
pushright(A::DenseMPO, B::SparseMPO, C::T, EnvL::SparseLeftEnvironmentTensor, i::Int64) where T <: Union{AdjointMPO,RefMPO} = pushright(A[i],B[i],C[i],EnvL)


pushleft(A::DenseMPO, B::DenseMPO, C::T, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) where T <: Union{AdjointMPO,RefMPO} = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPO, B::DenseMPO, C::T, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) where T <: Union{AdjointMPO,RefMPO} = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL.A))

pushleft(A::DenseMPS, B::DenseMPO, C::T₃, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) where T₃ <: Union{AdjointMPS,RefMPS} = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPS, B::DenseMPO, C::T₃, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) where T₃ <: Union{AdjointMPS,RefMPS} = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL.A))


pushleft(A::DenseMPO, B::AdjointMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPO, B::AdjointMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL.A))

pushleft(A::DenseMPO, B::RefMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(A[site], B[site], C[site], EnvR.A))
pushright(A::DenseMPO, B::RefMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(A[site], B[site], C[site], EnvL.A))

# layer 2 - dense

pushleft(A::DenseMPS, B::AdjointMPS, EnvR::DenseRightEnvironmentTensor{2}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvR.A))
pushright(A::DenseMPS, B::AdjointMPS, EnvL::DenseLeftEnvironmentTensor{2}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvL.A))

pushleft(A::DenseMPO, B::AdjointMPO, EnvR::DenseRightEnvironmentTensor{2}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvR.A))
pushright(A::DenseMPO, B::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{2}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvL.A))


