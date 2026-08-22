
#= TDVP =#

function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{2}, DenseMPOTensor{2}}, Alg::TDVPalgo,info::TDVPsweepinfo{L2R})
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1][site] = tl
    Env.layer[3][site] = tl'
    to = TimerOutput()
    EnvR = Env.envs[site+1]
    @timeit to "pushright" pushright!(Env)
    @timeit to "back evolve" ~, K = evolve!(tr, proj0(Env.envs[site+1],EnvR,issparse(Env.layer[2]) ? Env.layer[2][site+1].left : nothing;E₀ = info.E), -Alg.τ, Alg.solver)
    rmul!(tr,exp(Alg.τ * info.E))
    tr = splice(tr,Env.layer[1][site+1])
    Env.layer[1][site+1] = tr
    Env.layer[3][site+1] = tr'

    map(x -> Env.layer[x].center .+= 1,[1,3])
    return to,K
end

function pushleft!(Env::Environment{3}, tl::Union{MPSTensor{2}, DenseMPOTensor{2}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, Alg::TDVPalgo,info::TDVPsweepinfo{R2L})
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1][site] = tr
    Env.layer[3][site] = tr'
    to = TimerOutput()
    EnvL = Env.envs[site]
    @timeit to "pushleft" pushleft!(Env)
    @timeit to "back evolve" ~, K = evolve!(tl, proj0(EnvL,Env.envs[site],issparse(Env.layer[2]) ? Env.layer[2][site-1].right : nothing;E₀ = info.E), -Alg.τ, Alg.solver)
    rmul!(tl,exp(Alg.τ * info.E))
    tl = splice(Env.layer[1][site-1],tl)
    Env.layer[1][site-1] = tl
    Env.layer[3][site-1] = tl'

    map(x -> Env.layer[x].center .-= 1,[1,3])
    return to,K
end


function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, Alg::TDVPalgo,info::TDVPsweepinfo{L2R})
    @assert (site = Env.center[1] ) == Env.center[2]
    to = TimerOutput()
    Env.layer[1][site] = tl
    Env.layer[3][site] = tl'
    @timeit to "pushright!" pushright!(Env)
    @timeit to "back evolve" tr, K = evolve!(tr, proj1(Env,site+1;E₀ = info.E), -Alg.τ, Alg.solver)
    rmul!(tr,exp(Alg.τ * info.E))
    Env.layer[1][site+1] = tr
    Env.layer[3][site+1] = tr'

    map(x -> Env.layer[x].center .+= 1,[1,3])
    return to,K
end

function pushleft!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, Alg::TDVPalgo,info::TDVPsweepinfo{R2L})
    @assert (site = Env.center[1] ) == Env.center[2]
    to = TimerOutput()
    Env.layer[1][site] = tr
    Env.layer[3][site] = tr'
    @timeit to "pushleft!" pushleft!(Env)
    @timeit to "back evolve" tl, K = evolve!(tl, proj1(Env,site-1;E₀ = info.E), -Alg.τ, Alg.solver)
    rmul!(tl,exp(Alg.τ * info.E))
    Env.layer[1][site-1] = tl
    Env.layer[3][site-1] = tl'

    map(x -> Env.layer[x].center .-= 1,[1,3])
    return to,K
end
