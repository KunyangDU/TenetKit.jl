action(env::TaSKEnvironment) = action!(copy(env))
function action!(env::TaSKEnvironment{L,T}) where {L,T}
    reorthogonalize!(env)
    TC = Vector{T}(undef,L)
    for i in 1:L
        HL = proj1(env.OrthL[i], env.H[i], env.EnvR[i])
        HC = proj1(env.EnvL[i], env.H[i], env.EnvR[i])
        HR = proj1(env.EnvL[i], env.H[i], env.OrthR[i])
        TC[i] = action(HL,env.TR[i]) + action(HC,env.TC[i]) + action(HR,env.TL[i]) - env.Eg * env.TC[i]
    end 
    env.TC = TC
    orthogonalize!(env)   
    env.n += 1
    return env
end