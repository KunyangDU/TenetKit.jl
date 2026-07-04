function initialize!(env::TaSKEnvironment{L}) where L

    setdefault!(env)

    for i in 1:L-1; env.EnvL[i + 1] = pushright(env.TL[i], H[i], env.TL[i]', env.EnvL[i]); end
    for i in L:-1:2; env.EnvR[i - 1] = pushleft(env.TR[i], H[i], env.TR[i]', env.EnvR[i]); end
    
    # orthogonalize!(env)
    # reorthogonalize!(env)
end

function setdefault!(env::TaSKEnvironment{L,T,SparseMPO}) where {L,T}

    env.EnvL = Vector{SparseLeftEnvironmentTensor}(undef, L)
    env.EnvR = Vector{SparseRightEnvironmentTensor}(undef, L)
    env.OrthL = Vector{SparseLeftEnvironmentTensor}(undef, L)
    env.OrthR = Vector{SparseRightEnvironmentTensor}(undef, L)

    als = left_auxspace(env.TC[1])
    alr = right_auxspace(env.TC[end])

    env.EnvL[1] = SparseLeftEnvironmentTensor(ones(als,als))
    env.EnvR[end] = SparseRightEnvironmentTensor(ones(alr,alr))
    env.OrthL[1] = SparseLeftEnvironmentTensor(ones(als,als))
    env.OrthR[end] = SparseRightEnvironmentTensor(ones(alr,alr))

    return env
end

function initialize!(env::TaSKEnvironment{L}, A::T, O::SparseMPO{L}) where T <: Union{DenseMPS{L}, DenseMPO{L}} where L
    env′ = Environment([A,O,A'])
    initialize!(env′)
    for i in 1:L
        canonicalize!(env′,i)
        env.TC[i] = action(proj1(env′,i),A[i])
    end
    orthogonalize!(env)
    normalize!(env)
    return env
end