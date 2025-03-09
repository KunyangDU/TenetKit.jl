
function initialize!(env::Environment)
    env.envs = Vector{AbstractEnvironmentTensor}(undef, env.L + 1)
    setdefault!(env)
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


function setdefault!(env::Environment{3})
    if issparse(env.layer[2])
        env.envs[1] = SparseLeftEnvironmentTensor(isometry(reverse(map(x -> getAuxSpace(env.layer[x].ts[1])[1],[1,3]))...))
        env.envs[end] = SparseRightEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[end])[2],[1,3])...))
    else
        AuxSpaces = reverse(map(x -> getAuxSpace(env.layer[x].ts[1])[1],1:3))
        env.envs[1] = DenseLeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2] ⊗ AuxSpaces[3]))
        AuxSpaces = map(x -> getAuxSpace(env.layer[x].ts[end])[2],1:3)
        env.envs[end] = DenseRightEnvironmentTensor(isometry(AuxSpaces[1] ⊗ AuxSpaces[2], AuxSpaces[3]))
    end
end

function setdefault!(env::Environment{2})
    if !issparse(env.layer[1]) && !issparse(env.layer[2])
        env.envs[1] = DenseLeftEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[1])[1],1:2)...))
        env.envs[end] = DenseRightEnvironmentTensor(isometry(map(x -> getAuxSpace(env.layer[x].ts[end])[2],1:2)...))
    elseif issparse(env.layer[1]) && !issparse(env.layer[2])
        env.envs[1] = SparseLeftEnvironmentTensor(isometry((getAuxSpace(env.layer[2].ts[1])[1] |> y -> (trivial(y),y))...))
        env.envs[end] = SparseLeftEnvironmentTensor(isometry((getAuxSpace(env.layer[2].ts[end])[2] |> y -> (y,trivial(y)))...))
    end
end


