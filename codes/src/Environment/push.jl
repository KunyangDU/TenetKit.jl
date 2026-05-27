function pushleft!(env::Environment{R}) where R
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[2]] = pushleft(env.layer..., env.envs[env.center[2] + 1], env.center[2])

    env.center[2] -= 1
    ( env.center[1] > env.center[2] ) && ( env.center[1] -= 1 )
end

function pushright!(env::Environment{R}) where R
    @assert 1 ≤ env.center[1] ≤ env.center[2] ≤ env.L

    env.envs[env.center[1] + 1] = pushright(env.layer..., env.envs[env.center[1]], env.center[1])

    env.center[1] += 1
    ( env.center[1] > env.center[2] ) && ( env.center[2] += 1 )
end

function pushleft(A::AbstractMPS, mpo::SparseMPO, B::AbstractMPS, EnvR::SparseRightEnvironmentTensor{1}, site::Int64)
    @assert mpo.D[site][3] == EnvR.D[1]
    tmpEnvR = Vector{Any}(nothing, mpo.D[site][2])
    r_map = _validind1(mpo[site], R2L())
    validind = [(j, r_inds) for (j, r_inds) in enumerate(r_map) if !isempty(r_inds)]
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validind) && break
                j, r_inds = validind[ct]
                x = contract(A[site], mpo[site][j], B[site], sum(EnvR[r_inds]))
                lock(Lock)
                try
                    tmpEnvR[j] = axpy!(1, x, tmpEnvR[j])
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for (j, r_inds) in validind
            tmpEnvR[j] = axpy!(1, contract(A[site], mpo[site][j], B[site], sum(EnvR[r_inds])), tmpEnvR[j])
        end
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor}, tmpEnvR))
end

function pushright(A::AbstractMPS, mpo::SparseMPO, B::AbstractMPS, EnvL::SparseLeftEnvironmentTensor{1}, site::Int64)
    @assert mpo.D[site][1] == EnvL.D[1]
    tmpEnvL = Vector{Any}(nothing, mpo.D[site][2])
    l_map = _validind1(mpo[site], L2R())
    validind = [(j, l_inds) for (j, l_inds) in enumerate(l_map) if !isempty(l_inds)]
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validind) && break
                j, l_inds = validind[ct]
                x = contract(A[site], mpo[site][j], B[site], sum(EnvL[l_inds]))
                lock(Lock)
                try
                    tmpEnvL[j] = axpy!(1, x, tmpEnvL[j])
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for (j, l_inds) in validind
            tmpEnvL[j] = axpy!(1, contract(A[site], mpo[site][j], B[site], sum(EnvL[l_inds])), tmpEnvL[j])
        end
    end
    return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor}, tmpEnvL))
end

# function pushright!(env::Environment{N}, tl::DenseMPOTensor{4}, tr::DenseMPOTensor{4}) where N
#     @show "test"
#     @assert (site = env.center[1] ) == env.center[2]
#     env.layer[end][site:site+1] = map(adjoint,[tl,tr])
#     env.layer[end].center = env.center
#     map(v -> canonicalize!(env.layer[v],site + 1),1:N-1)
#     pushright!(env)
# end

# function pushleft!(env::Environment{N}, tl::DenseMPOTensor{4}, tr::DenseMPOTensor{4}) where N
#     @assert (site = env.center[1] ) == env.center[2]
#     env.layer[end][site-1:site] = map(adjoint,[tl,tr])
#     env.layer[end].center = env.center
#     map(v -> canonicalize!(env.layer[v],site - 1),1:N-1)
#     pushleft!(env)
# end

pushleft(A::DenseMPO, B::AdjointMPO, EnvR::DenseRightEnvironmentTensor{2}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvR.A))
pushright(A::DenseMPO, B::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{2}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvL.A))

pushleft(A::DenseMPO, B::DenseMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPO, B::DenseMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL.A))

pushleft(A::DenseMPO, B::SparseMPO, C::T, EnvR::SparseRightEnvironmentTensor, site::Int64) where T <: Union{AdjointMPO,RefMPO} = SparseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR))
pushright(A::DenseMPO, B::SparseMPO, C::T, EnvL::SparseLeftEnvironmentTensor, site::Int64) where T <: Union{AdjointMPO,RefMPO} = SparseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL))

pushleft(A::DenseMPS, B::AdjointMPS, EnvR::DenseRightEnvironmentTensor{2}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvR.A))

#= Env4 =#

function pushright(Hup::SparseMPO, ρ::DenseMPO, Hdown::SparseMPO, ρ′::AdjointMPO, EnvL::SparseLeftEnvironmentTensor{2}, site::Int64)
    tmpEnvL = Array{Any}(nothing, Hup.D[site][2], Hdown.D[site][2])
    l_map_up = _validind1(Hup[site], L2R())
    l_map_down = _validind1(Hdown[site], L2R())
    vind_up = [(j, l_inds) for (j, l_inds) in enumerate(l_map_up) if !isempty(l_inds)]
    vind_down = [(j, l_inds) for (j, l_inds) in enumerate(l_map_down) if !isempty(l_inds)]
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        pairs = [(a,b) for a in eachindex(vind_up) for b in eachindex(vind_down)]
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(pairs) && break
                a, b = pairs[ct]
                op_up, l_up = vind_up[a]
                op_down, l_down = vind_down[b]
                for i in l_up, j in l_down
                    C = pushright(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], EnvL.A[i,j])
                    lock(Lock)
                    try
                        tmpEnvL[op_up,op_down] = axpy!(1, C, tmpEnvL[op_up,op_down])
                    catch
                        rethrow()
                    finally
                        unlock(Lock)
                    end
                end
            end
        end
    else
        for (op_up, l_up) in vind_up, (op_down, l_down) in vind_down
            for i in l_up, j in l_down
                C = pushright(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], EnvL.A[i,j])
                tmpEnvL[op_up,op_down] = axpy!(1, C, tmpEnvL[op_up,op_down])
            end
        end
    end
    return SparseLeftEnvironmentTensor(convert(Array{LeftEnvironmentTensor}, tmpEnvL))
end

function pushleft(Hup::SparseMPO, ρ::DenseMPO, Hdown::SparseMPO, ρ′::AdjointMPO, EnvR::SparseRightEnvironmentTensor{2}, site::Int64)
    tmpEnvR = Array{Any}(nothing, Hup.D[site][2], Hdown.D[site][2])
    r_map_up = _validind1(Hup[site], R2L())
    r_map_down = _validind1(Hdown[site], R2L())
    vind_up = [(j, r_inds) for (j, r_inds) in enumerate(r_map_up) if !isempty(r_inds)]
    vind_down = [(j, r_inds) for (j, r_inds) in enumerate(r_map_down) if !isempty(r_inds)]
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        pairs = [(a,b) for a in eachindex(vind_up) for b in eachindex(vind_down)]
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(pairs) && break
                a, b = pairs[ct]
                op_up, r_up = vind_up[a]
                op_down, r_down = vind_down[b]
                for k in r_up, l in r_down
                    C = pushleft(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], EnvR.A[k,l])
                    lock(Lock)
                    try
                        tmpEnvR[op_up,op_down] = axpy!(1, C, tmpEnvR[op_up,op_down])
                    catch
                        rethrow()
                    finally
                        unlock(Lock)
                    end
                end
            end
        end
    else
        for (op_up, r_up) in vind_up, (op_down, r_down) in vind_down
            for k in r_up, l in r_down
                C = pushleft(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], EnvR.A[k,l])
                tmpEnvR[op_up,op_down] = axpy!(1, C, tmpEnvR[op_up,op_down])
            end
        end
    end
    return SparseRightEnvironmentTensor(convert(Array{RightEnvironmentTensor}, tmpEnvR))
end

pushright(::Nothing, A::DenseMPOTensor{4}, h::AbstractLocalOperator, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2}) = contract(A,h,A′,EnvL)
function pushright(h::LocalOperator{1, 1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ h.A[1,5] * A.A[4,2,-2,1] * A′.A[-1,5,4,3] * EnvL.A[3,2]
    return LeftEnvironmentTensor(tmp)
end
function pushright(::IdentityOperator{1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,1,-2,3] * A′.A[-1,3,4,2] * EnvL.A[2,1]
    return LeftEnvironmentTensor(tmp)
end
pushleft(::Nothing, A::DenseMPOTensor{4}, h::AbstractLocalOperator, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2}) = contract(A,h,A′,EnvR)
function pushleft(h::LocalOperator{1, 1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ h.A[1,4] * A.A[5,-1,2,1] * A′.A[3,4,5,-2] * EnvR.A[2,3]
    return RightEnvironmentTensor(tmp)
end
function pushleft(::IdentityOperator{1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,-1,1,3] * A′.A[2,3,4,-2] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end
##
function pushleft(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor x[-1;-2] ≔ ht.A[1,6] * objt.A[2,-1,3,1] * hb.A[5,2] * objb.A[4,6,5,-2] * EnvR.A[3,4]
    return RightEnvironmentTensor(x)
end

function pushleft(::IdentityOperator{1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    # @tensor x[-1;-2] ≔ objt.A[2,-1,3,1] * hb.A[5,2] * objb.A[4,1,5,-2] * EnvR.A[3,4]
    @tensor x[-1;-2] ≔ objt.A[1,-1,2,4] * hb.A[5,1] * objb.A[3,4,5,-2] * EnvR.A[2,3]
    return RightEnvironmentTensor(x)
end

function pushleft(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    # @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[2,-1,3,1] * objb.A[4,5,2,-2] * EnvR.A[3,4]
    @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[4,-1,2,1] * objb.A[3,5,4,-2] * EnvR.A[2,3]
    return RightEnvironmentTensor(x)
end

function pushleft(::IdentityOperator{1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor x[-1;-2] ≔ objt.A[3,-1,1,4] * objb.A[2,4,3,-2] * EnvR.A[1,2]
    return RightEnvironmentTensor(x)
end

#= TDVP =#

# function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{2}, DenseMPOTensor{2}}, Alg::TDVPalgo,info::TDVPsweepinfo{L2R})
#     @assert (site = Env.center[1] ) == Env.center[2]
#     Env.layer[1][site] = tl
#     Env.layer[3][site] = tl'
#     to = TimerOutput()
#     @timeit to "pushright" pushright!(Env)
#     @timeit to "back evolve" ~, K = evolve!(tr, projleft0(Env), -Alg.τ)
#     tr = contract(tr,Env.layer[1][site+1])
#     Env.layer[1][site+1] = tr
#     Env.layer[3][site+1] = tr'

#     map(x -> Env.layer[x].center .+= 1,[1,3])
#     return to,K
# end

# function pushleft!(Env::Environment{3}, tl::Union{MPSTensor{2}, DenseMPOTensor{2}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, Alg::TDVPalgo,info::TDVPsweepinfo{L2R})
#     @assert (site = Env.center[1] ) == Env.center[2]
#     Env.layer[1][site] = tr
#     Env.layer[3][site] = tr'
#     to = TimerOutput()
#     @timeit to "pushleft" pushleft!(Env)
#     @timeit to "back evolve" ~, K = evolve!(tl, projright0(Env), -Alg.τ)
#     tl = contract(Env.layer[1][site-1],tl)
#     Env.layer[1][site-1] = tl
#     Env.layer[3][site-1] = tl'

#     map(x -> Env.layer[x].center .-= 1,[1,3])
#     return to,K
# end


# function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, Alg::TDVPalgo,info::TDVPsweepinfo{L2R})
#     @assert (site = Env.center[1] ) == Env.center[2]
#     to = TimerOutput()
#     Env.layer[1][site] = tl
#     Env.layer[3][site] = adjoint(tl)
#     @timeit to "pushright!" pushright!(Env)
#     @timeit to "back evolve" tr, K = evolve!(tr, proj1(Env,site+1), -Alg.τ)
#     Env.layer[1][site+1] = tr
#     Env.layer[3][site+1] = adjoint(tr)

#     map(x -> Env.layer[x].center .+= 1,[1,3])
#     return to,K
# end

# function pushleft!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{3}, DenseMPOTensor{4}}, Alg::TDVPalgo,info::TDVPsweepinfo{L2R})
#     @assert (site = Env.center[1] ) == Env.center[2]
#     to = TimerOutput()
#     Env.layer[1][site] = tr
#     Env.layer[3][site] = adjoint(Env.layer[1][site])
#     @timeit to "pushleft!" pushleft!(Env)
#     @timeit to "back evolve" tl, K = evolve!(tl, proj1(Env,site-1), -Alg.τ)
#     Env.layer[1][site-1] = tl
#     Env.layer[3][site-1] = adjoint(Env.layer[1][site-1])

#     map(x -> Env.layer[x].center .-= 1,[1,3])
#     return to,K
# end

#= -------- =#

function pushright!(Env::Environment{3}, tl::Union{MPSTensor{3}, DenseMPOTensor{4}}, tr::Union{MPSTensor{2}, DenseMPOTensor{2}}, Alg::TDVPalgo,info::TDVPsweepinfo{L2R})
    @assert (site = Env.center[1] ) == Env.center[2]
    Env.layer[1][site] = tl
    Env.layer[3][site] = tl'
    to = TimerOutput()
    @timeit to "pushright" pushright!(Env)
    @timeit to "back evolve" ~, K = evolve!(tr, projleft0(Env;E₀ = info.E), -Alg.τ, Alg.solver)
    rmul!(tr,exp(Alg.τ * info.E))
    tr = contract(tr,Env.layer[1][site+1])
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
    @timeit to "pushleft" pushleft!(Env)
    @timeit to "back evolve" ~, K = evolve!(tl, projright0(Env;E₀ = info.E), -Alg.τ, Alg.solver)
    rmul!(tl,exp(Alg.τ * info.E))
    tl = contract(Env.layer[1][site-1],tl)
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

#= densify! =#

function pushleft(A::SparseMPO, B::AdjointMPO, EnvR::SparseRightEnvironmentTensor{1}, site::Int64)
    @assert A.D[site][3] == EnvR.D[1]
    tmpEnvR = Vector{Any}(nothing, A.D[site][2])
    r_map = _validind1(A[site], R2L())
    for (j, r_inds) in enumerate(r_map)
        isempty(r_inds) && continue
        tmpEnvR[j] = axpy!(1, contract(A[site][j], B[site], sum(EnvR[r_inds])), tmpEnvR[j])
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor}, tmpEnvR))
end

function pushright(A::SparseMPO, B::AdjointMPO, EnvL::SparseLeftEnvironmentTensor{1}, site::Int64)
    @assert A.D[site][1] == EnvL.D[1]
    tmpEnvL = Vector{Any}(nothing, A.D[site][2])
    l_map = _validind1(A[site], L2R())
    for (j, l_inds) in enumerate(l_map)
        isempty(l_inds) && continue
        tmpEnvL[j] = axpy!(1, contract(A[site][j], B[site], sum(EnvL[l_inds])), tmpEnvL[j])
    end
    return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor}, tmpEnvL))
end


pushleft(A::DenseMPO, B::AdjointMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPO, B::AdjointMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL.A))

pushleft(A::DenseMPO, B::RefMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(A[site], B[site], C[site], EnvR.A))
pushright(A::DenseMPO, B::RefMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(A[site], B[site], C[site], EnvL.A))
