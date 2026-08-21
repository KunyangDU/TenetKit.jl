
function Base.replace!(C::DenseMPO,A::DenseMPO)
    @assert C.L == A.L
    C.ts = A.ts
    C.center = A.center
    return C
end

function tr(ρ::DenseMPO)
    ρ′ = ρ'
    result = tr(ρ, ρ′)
    cleanup!(ρ′)
    return result
end

function tr(ρ1::DenseMPO,ρ2::AdjointMPO)
    Env = Environment([ρ1,ρ2])
    initialize!(Env)
    try return _scalar(Env) finally cleanup!(Env) end
end

function tr(ρ::DenseMPO, A::SparseMPO)
    ρ′ = ρ'
    Env = Environment([ρ, A, ρ′])
    initialize!(Env)
    try return _scalar(Env) finally cleanup!(Env); cleanup!(ρ′) end
end

tr1(obj::DenseMPO) = norm(obj[1])

"""
compatible for N-layer Environment
"""
# function _scalar(env::Environment{N}) where N
#     @assert (site = env.center[1]) == env.center[2]
#     return contract(env.envs[site],map(x -> env.layer[x][site], 1:length(env.layer))...,env.envs[site+1])
# end

function _scalar(EnvL::LeftEnvironmentTensor{2})
    return @tensor EnvL.A[1,1]
end

function _scalar(env::Environment{2})
    @assert (site = env.center[1]) == env.center[2]
    return inner(env.layer[2][site], action(proj1(env, site), env.layer[1][site]))
end

function _scalar(env::Environment{3})
    @assert (site = env.center[1]) == env.center[2]
    return inner(env.layer[3][site], action(proj1(env, site), env.layer[1][site]))
end


