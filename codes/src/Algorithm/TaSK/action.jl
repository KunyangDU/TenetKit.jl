action(env::TaSKEnvironment) = action!(copy(env))
function action!(env::TaSKEnvironment{L,T}) where {L,T}
    to = TimerOutput()
    @timeit to "reorthogonalize!" reorthogonalize!(env)
    TC = Vector{T}(undef,L)
    for i in 1:L
        HL = proj1(env.OrthL[i], env.H[i], env.EnvR[i])
        HC = proj1(env.EnvL[i], env.H[i], env.EnvR[i])
        HR = proj1(env.EnvL[i], env.H[i], env.OrthR[i])
        @timeit to "update!" TC[i] = action(HL,env.TR[i]) + action(HC,env.TC[i]) + action(HR,env.TL[i]) - env.Eg * env.TC[i]
        merge!(to,get_timer("action");tree_point = ["update!"])
        reset_timer!(get_timer("action"))
    end 
    env.TC = TC
    @timeit to "orthogonalize!" orthogonalize!(env)   
    env.n += 1
    return env,to
end