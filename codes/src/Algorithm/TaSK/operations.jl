function initialize!(env::TaSKEnvironment{L}) where L

    setdefault!(env)

    for i in 1:L-1; env.EnvL[i + 1] = pushright(env.TL[i], env.H[i], env.TL[i]', env.EnvL[i]); end
    for i in L:-1:2; env.EnvR[i - 1] = pushleft(env.TR[i], env.H[i], env.TR[i]', env.EnvR[i]); end

end

function setdefault!(env::TaSKEnvironment{L,T,SparseMPO}) where {L,T}

    env.EnvL = Vector{SparseLeftEnvironmentTensor}(undef, L)
    env.EnvR = Vector{SparseRightEnvironmentTensor}(undef, L)
    env.OrthL = Vector{SparseLeftEnvironmentTensor}(undef, L)
    env.OrthR = Vector{SparseRightEnvironmentTensor}(undef, L)

    als = left_auxspace(env.TL[1])
    alr = right_auxspace(env.TR[end])

    env.EnvL[1] = SparseLeftEnvironmentTensor(ones(als,als))
    env.EnvR[end] = SparseRightEnvironmentTensor(ones(alr,alr))

    als_u = left_auxspace(env.TR[1])
    als_d = left_auxspace(env.TL[1])
    alr_u = right_auxspace(env.TL[end])
    alr_d = right_auxspace(env.TR[end])

    env.OrthL[1] = SparseLeftEnvironmentTensor(ones(als_d,als_u))
    env.OrthR[end] = SparseRightEnvironmentTensor(ones(alr_u,alr_d))

    return env
end

function initialize!(env::TaSKEnvironment{L}, A::T, O::SparseMPO{L}) where T <: Union{DenseMPS{L}, DenseMPO{L}} where L
    env′ = Environment([A,O,A'])
    initialize!(env′)
    for i in 1:L
        canonicalize!(env′.layer[1],i)
        canonicalize!(env′.layer[3],i)
        canonicalize!(env′,i)
        env.TC[i] = action(proj1(env′,i),A[i])
    end
    canonicalize!(env′.layer[1],1)
    canonicalize!(env′.layer[3],1)
    orthogonalize!(env)
    return env
end