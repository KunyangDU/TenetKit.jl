reorthogonalize!(env::TaSKEnvironment) = (leftorthogonalize!(env); rightorthogonalize!(env))

leftorthogonalize!(env::TaSKEnvironment{L}) where L = for i in 1:L-1; env.OrthL[i + 1] = pushright(env.TC[i], H[i], env.TL[i]', env.EnvL[i]) + pushright(env.TR[i], H[i], env.TL[i]', env.OrthL[i]); end
rightorthogonalize!(env::TaSKEnvironment{L}) where L = for i in L:-1:2; env.OrthR[i - 1] = pushleft(env.TC[i], H[i], env.TR[i]', env.EnvR[i]) + pushleft(env.TL[i], H[i], env.TR[i]', env.OrthR[i]); end

function orthogonalize!(env::TaSKEnvironment{L,MPSTensor{3}}, i::Int64) where L
    @tensor tmp[-1 -2;-3] ≔ env.TC[i].A[1,2,-3] * env.TL[i].A'[3,1,2] * env.TL[i].A[-1,-2,3] 
    env.TC[i] -= MPSTensor(tmp)
    return env
end
orthogonalize!(env::TaSKEnvironment{L}) where L = (for i in 1:L; orthogonalize!(env,i); end; env)
