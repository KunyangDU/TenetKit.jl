
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

function tr(ρ::DenseMPO, Opr::SparseMPO)
    Env = Environment([deepcopy(ρ), Opr, ρ'])
    initialize!(Env)
    return _scalar(Env)
end

"""
compatible for N-layer Environment
"""
function _scalar(Env::Environment{N}) where N
    @assert (site = Env.center[1]) == Env.center[2]
    t1 = map(x -> Env.layer[x].ts[site], 1:length(Env.layer))
    tmp = contract(Env.envs[site],t1...,Env.envs[site+1])
    return tmp
end

function scalar(Env::Environment{3})
    @assert Env.center[1] == Env.center[2]
    contract(Env.layer[3].ts[Env.center[1]], action(proj1(Env, Env.center[1]), Env.layer[1].ts[Env.center[1]]))
end



