
function initialize!(env::Environment;kwargs...)
    env.envs = Vector{AbstractEnvironmentTensor}(undef, env.L + 1)
    setdefault!(env;kwargs...)
    canonicalize!(env,1)
end

function canonicalize!(env::Environment,sl::Int64,sr::Int64)
    @assert 1 ≤ sl ≤ sr ≤ env.L + 1

    for _ in env.center[1]:sl-1
        pushright!(env)
    end

    for _ in env.center[2]:-1:sr+1
        pushleft!(env)
    end

end

function canonicalize!(env::Environment,si::Int64)
    @assert 1 ≤ si ≤ env.L + 1
    canonicalize!(env,si,si)
end


function setdefault!(env::Environment{3};kwargs...)
    if issparse(env.layer[2])
        lds = get(kwargs,:left_default_space, reverse(map(x -> getAuxSpace(env.layer[x].ts[1])[1],[1,3])))
        rds = get(kwargs,:right_default_space, map(x -> getAuxSpace(env.layer[x].ts[end])[2],[1,3]))
        env.envs[1] = SparseLeftEnvironmentTensor(isometry(lds...))
        env.envs[end] = SparseRightEnvironmentTensor(isometry(rds...))
    else
        AuxSpaces = reverse(map(x -> getAuxSpace(env.layer[x].ts[1])[1],1:3))
        env.envs[1] = DenseLeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2] ⊗ AuxSpaces[3]))
        AuxSpaces = map(x -> getAuxSpace(env.layer[x].ts[end])[2],1:3)
        env.envs[end] = DenseRightEnvironmentTensor(isometry(AuxSpaces[1] ⊗ AuxSpaces[2], AuxSpaces[3]))
    end
end

function setdefault!(env::Environment{2};kwargs...)
    if !issparse(env.layer[1]) && !issparse(env.layer[2])
        env.envs[1] = DenseLeftEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[1])[1],1:2)...))
        env.envs[end] = DenseRightEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[end])[2],1:2)...))
    elseif issparse(env.layer[1]) && !issparse(env.layer[2])
        lds = get(kwargs,:left_default_space, getAuxSpace(env.layer[2].ts[1])[1] |> y -> (trivial(y),y))
        rds = get(kwargs,:right_default_space, getAuxSpace(env.layer[2].ts[end])[2] |> y -> (y,trivial(y)))
        env.envs[1] = SparseLeftEnvironmentTensor(isometry(lds...))
        env.envs[end] = SparseRightEnvironmentTensor(isometry(rds...))
    elseif !issparse(env.layer[1]) && issparse(env.layer[2])
        lds = get(kwargs,:left_default_space, repeat([getAuxSpace(env.layer[1].ts[1])[1],],2))
        rds = get(kwargs,:right_default_space, repeat([getAuxSpace(env.layer[1].ts[end])[2],],2))
        env.envs[1] = SparseLeftEnvironmentTensor(isometry(lds...))
        env.envs[end] = SparseRightEnvironmentTensor(isometry(rds...))
    end
end

function setdefault!(Env::Environment{4})
    if issparse(Env.layer[1]) && issparse(Env.layer[3])
        ρ1 = Env.layer[2]
        ρ2′ = Env.layer[4]
        tmpL = (isometry(space(ρ2′.ts[1])[4]', space(ρ1.ts[1])[2]))
        tmpR = (isometry(space(ρ1.ts[end])[3]', space(ρ2′.ts[end])[1]))
        Env.envs[1] = SparseLeftEnvironmentTensor([tmpL;;])
        Env.envs[end] = SparseRightEnvironmentTensor([tmpR;;])
    end
end

