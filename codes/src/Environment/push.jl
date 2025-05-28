function pushleft!(env::Environment{R}) where R
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[2]] = pushleft(env.layer..., env.envs[env.center[2] + 1], env.center[2])

    env.center[2] -= 1
    ( env.center[1] > env.center[2] ) && ( env.center[1] -= 1 )
end

function pushleft(A::AbstractMPS, mpo::SparseMPO, B::AbstractMPS, EnvR::SparseRightEnvironmentTensor, site::Int64)
    @assert mpo.D[site][2] == EnvR.D
    tmpEnvR = Vector{Any}(nothing,mpo.D[site][1])
    for i in eachindex(tmpEnvR), j in 1:EnvR.D
        isnothing(mpo.ts[site].m[i,j]) && continue
        if isnothing(tmpEnvR[i])
            tmpEnvR[i] = contract(A.ts[site], mpo.ts[site].m[i,j], B.ts[site], EnvR.A[j])
        else 
            tmpEnvR[i] += contract(A.ts[site], mpo.ts[site].m[i,j], B.ts[site], EnvR.A[j])
        end
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor},tmpEnvR))
end


function pushright!(env::Environment{R}) where R
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[1] + 1] = pushright(env.layer..., env.envs[env.center[1]], env.center[1])

    env.center[1] += 1
    ( env.center[1] > env.center[2] ) && ( env.center[2] += 1 )
end

function pushright(A::AbstractMPS, mpo::SparseMPO, B::AbstractMPS, EnvL::SparseLeftEnvironmentTensor, site::Int64)
    @assert mpo.D[site][1] == EnvL.D
    tmpEnvL = Vector{Any}(nothing,mpo.D[site][2])
    for i in eachindex(tmpEnvL), j in 1:EnvL.D
        isnothing(mpo.ts[site].m[j,i]) && continue
        if isnothing(tmpEnvL[i])
            tmpEnvL[i] = contract(A.ts[site], mpo.ts[site].m[j,i], B.ts[site],EnvL.A[j])
        else 
            tmpEnvL[i] += contract(A.ts[site], mpo.ts[site].m[j,i], B.ts[site],EnvL.A[j])
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor},tmpEnvL))
end

# function pushright!(env::Environment{N}, tl::DenseMPOTensor{4}, tr::DenseMPOTensor{4}) where N
#     @show "test"
#     @assert (site = env.center[1] ) == env.center[2]
#     env.layer[end].ts[site:site+1] = map(adjoint,[tl,tr])
#     env.layer[end].center = env.center
#     map(v -> canonicalize!(env.layer[v],site + 1),1:N-1)
#     pushright!(env)
# end

# function pushleft!(env::Environment{N}, tl::DenseMPOTensor{4}, tr::DenseMPOTensor{4}) where N
#     @assert (site = env.center[1] ) == env.center[2]
#     env.layer[end].ts[site-1:site] = map(adjoint,[tl,tr])
#     env.layer[end].center = env.center
#     map(v -> canonicalize!(env.layer[v],site - 1),1:N-1)
#     pushleft!(env)
# end

pushleft(A::DenseMPO, B::AdjointMPO, EnvR::DenseRightEnvironmentTensor{2}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x.ts[site],(A,B))..., EnvR.A))
pushleft(A::DenseMPO, B::DenseMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x.ts[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPO, B::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{2}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x.ts[site],(A,B))..., EnvL.A))
pushright(A::DenseMPO, B::DenseMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x.ts[site],(A,B,C))..., EnvL.A))
pushleft(A::DenseMPO, B::SparseMPO, C::AdjointMPO, EnvR::SparseRightEnvironmentTensor, site::Int64) = SparseRightEnvironmentTensor(contract(map(x -> x.ts[site],(A,B,C))..., EnvR))
pushright(A::DenseMPO, B::SparseMPO, C::AdjointMPO, EnvL::SparseLeftEnvironmentTensor, site::Int64) = SparseLeftEnvironmentTensor(contract(map(x -> x.ts[site],(A,B,C))..., EnvL))
pushleft(A::DenseMPS, B::AdjointMPS, EnvR::DenseRightEnvironmentTensor{2}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x.ts[site],(A,B))..., EnvR.A))

#= TDVP =#

function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{2}, DenseMPOTensor{2}}, τ::Number)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site] = tl
    Env.layer[3].ts[site] = tl'
    to = TimerOutput()
    @timeit to "pushright" pushright!(Env)
    @timeit to "back evolve" ~, K = evolve!(tr, projleft0(Env), -τ)
    tr = contract(tr,Env.layer[1].ts[site+1])
    Env.layer[1].ts[site+1] = tr
    Env.layer[3].ts[site+1] = tr'

    map(x -> Env.layer[x].center .+= 1,[1,3])
    return to,K
end

function pushleft!(Env::Environment{3}, tl::Union{MPSTensor{2}, DenseMPOTensor{2}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, τ::Number)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site] = tr
    Env.layer[3].ts[site] = tr'
    to = TimerOutput()
    @timeit to "pushleft" pushleft!(Env)
    @timeit to "back evolve" ~, K = evolve!(tl, projright0(Env), -τ)
    tl = contract(Env.layer[1].ts[site-1],tl)
    Env.layer[1].ts[site-1] = tl
    Env.layer[3].ts[site-1] = tl'

    map(x -> Env.layer[x].center .-= 1,[1,3])
    return to,K
end


function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, τ::Number)
    @assert (site = Env.center[1] ) == Env.center[2]
    to = TimerOutput()
    Env.layer[1].ts[site] = tl
    Env.layer[3].ts[site] = adjoint(tl)
    @timeit to "pushright!" pushright!(Env)
    @timeit to "back evolve" tr, K = evolve!(tr, proj1(Env,site+1), -τ)
    Env.layer[1].ts[site+1] = tr
    Env.layer[3].ts[site+1] = adjoint(tr)

    map(x -> Env.layer[x].center .+= 1,[1,3])
    return to,K
end

function pushleft!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, τ::Number)
    @assert (site = Env.center[1] ) == Env.center[2]
    to = TimerOutput()
    Env.layer[1].ts[site] = tr
    Env.layer[3].ts[site] = adjoint(Env.layer[1].ts[site])
    @timeit to "pushleft!" pushleft!(Env)
    @timeit to "back evolve" tl, K = evolve!(tl, proj1(Env,site-1), -τ)
    Env.layer[1].ts[site-1] = tl
    Env.layer[3].ts[site-1] = adjoint(Env.layer[1].ts[site-1])

    map(x -> Env.layer[x].center .-= 1,[1,3])
    return to,K
end

#= -------- =#

function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{2}, DenseMPOTensor{2}}, τ::Number, alg::Krylovalgo)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site] = tl
    Env.layer[3].ts[site] = tl'
    to = TimerOutput()
    @timeit to "pushright" pushright!(Env)
    @timeit to "back evolve" ~, K = evolve!(tr, projleft0(Env), -τ, alg)
    tr = contract(tr,Env.layer[1].ts[site+1])
    Env.layer[1].ts[site+1] = tr
    Env.layer[3].ts[site+1] = tr'

    map(x -> Env.layer[x].center .+= 1,[1,3])
    return to,K
end

function pushleft!(Env::Environment{3}, tl::Union{MPSTensor{2}, DenseMPOTensor{2}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, τ::Number, alg::Krylovalgo)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site] = tr
    Env.layer[3].ts[site] = tr'
    to = TimerOutput()
    @timeit to "pushleft" pushleft!(Env)
    @timeit to "back evolve" ~, K = evolve!(tl, projright0(Env), -τ, alg)
    tl = contract(Env.layer[1].ts[site-1],tl)
    Env.layer[1].ts[site-1] = tl
    Env.layer[3].ts[site-1] = tl'

    map(x -> Env.layer[x].center .-= 1,[1,3])
    return to,K
end


function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, τ::Number, alg::Krylovalgo)
    @assert (site = Env.center[1] ) == Env.center[2]
    to = TimerOutput()
    Env.layer[1].ts[site] = tl
    Env.layer[3].ts[site] = adjoint(tl)
    @timeit to "pushright!" pushright!(Env)
    @timeit to "back evolve" tr, K = evolve!(tr, proj1(Env,site+1), -τ, alg)
    Env.layer[1].ts[site+1] = tr
    Env.layer[3].ts[site+1] = adjoint(tr)

    map(x -> Env.layer[x].center .+= 1,[1,3])
    return to,K
end

function pushleft!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, τ::Number, alg::Krylovalgo)
    @assert (site = Env.center[1] ) == Env.center[2]
    to = TimerOutput()
    Env.layer[1].ts[site] = tr
    Env.layer[3].ts[site] = adjoint(Env.layer[1].ts[site])
    @timeit to "pushleft!" pushleft!(Env)
    @timeit to "back evolve" tl, K = evolve!(tl, proj1(Env,site-1), -τ, alg)
    Env.layer[1].ts[site-1] = tl
    Env.layer[3].ts[site-1] = adjoint(Env.layer[1].ts[site-1])

    map(x -> Env.layer[x].center .-= 1,[1,3])
    return to,K
end

#= DMRG =#

function pushright!(Env::Environment{3},tl::MPSTensor, tr::MPSTensor)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site:site+1] = [tl,tr]
    Env.layer[3].ts[site:site+1] = adjoint(Env.layer[1].ts[site:site+1])
    pushright!(Env)
    map(x -> Env.layer[x].center .+= 1,[1,3])
end

function pushleft!(Env::Environment{3},tl::MPSTensor, tr::MPSTensor)
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1].ts[site-1:site] = [tl, tr]
    Env.layer[3].ts[site-1:site] = adjoint(Env.layer[1].ts[site-1:site])
    pushleft!(Env)
    map(x -> Env.layer[x].center .-= 1,[1,3])
end



