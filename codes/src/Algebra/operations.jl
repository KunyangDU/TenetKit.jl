
function Base.replace!(C::DenseMPO,A::DenseMPO)
    @assert C.L == A.L
    C.ts = A.ts
    C.center = A.center
    return C
end

function tr(ρ::DenseMPO)
    return tr(ρ,ρ')
end

function tr(ρ1::DenseMPO,ρ2::AdjointMPO)
    Env = Environment([deepcopy(ρ1),deepcopy(ρ2)])
    initialize!(Env)
    return _scalar(Env)
end

function tr(ρ::DenseMPO, A::SparseMPO)
    Env = Environment([deepcopy(ρ), A, ρ'])
    initialize!(Env)
    return _scalar(Env)
end

tr1(obj::DenseMPO) = norm(obj.ts[1])

"""
compatible for N-layer Environment
"""
function _scalar(Env::Environment{3})
    @assert Env.center[1] == Env.center[2]
    return contract(Env.layer[3].ts[Env.center[1]], action(proj1(Env, Env.center[1]), Env.layer[1].ts[Env.center[1]]))
end

function _scalar(Env::Environment{N}) where N
    @assert (site = Env.center[1]) == Env.center[2]
    t1 = map(x -> Env.layer[x].ts[site], 1:length(Env.layer))
    tmp = contract(Env.envs[site],t1...,Env.envs[site+1])
    return tmp
end

function _scalar(EnvL::LeftEnvironmentTensor{2})
    return @tensor EnvL.A[1,1]
end

function _scalar(env::Environment{2})
    ans = 0.0
    if env.center[1] ≠ env.center[2]
        env.envs = CachedVector{AbstractEnvironmentTensor}(env.L + 1, _cache_memory_limit(AbstractEnvironmentTensor))
        setdefault!(env)
        envL,envR = env.envs[1],env.envs[end]
        for i in env.L:-1:1
            envR = SparseRightEnvironmentTensor(contract(env.layer[1].ts[i],env.layer[2].ts[i],env.layer[1].ts[i]',envR))
        end
        for i in eachindex(envL.A)
            ans += @tensor envL.A[i].A[1,2] * envR.A[i].A[2,1]
        end
    else
        site = env.center[1]
        ans = contract(env.envs[site],env.layer[1].ts[site],env.layer[2].ts[site],env.envs[site+1])
    end
    return ans
end


# function scalar(Env::Environment{3})
#     @assert Env.center[1] == Env.center[2]
#     contract(Env.layer[3].ts[Env.center[1]], action(proj1(Env, Env.center[1]), Env.layer[1].ts[Env.center[1]]))
# end



