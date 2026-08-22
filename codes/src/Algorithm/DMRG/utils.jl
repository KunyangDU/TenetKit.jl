
#= DMRG =#

function pushright!(Env::Environment{3},tl::MPSTensor, tr::MPSTensor)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1][site:site+1] = [tl,tr]
    Env.layer[3][site:site+1] = adjoint(Env.layer[1][site:site+1])
    pushright!(Env)
    map(x -> Env.layer[x].center .+= 1,[1,3])
end

function pushleft!(Env::Environment{3},tl::MPSTensor, tr::MPSTensor)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1][site-1:site] = [tl, tr]
    Env.layer[3][site-1:site] = adjoint(Env.layer[1][site-1:site])
    pushleft!(Env)
    map(x -> Env.layer[x].center .-= 1,[1,3])
end
