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

pushleft(A::DenseMPS, mpo::SparseMPO, B::T, EnvR::SparseRightEnvironmentTensor{1}, i::Int64) where T <: Union{AdjointMPS,RefMPS} = pushleft(A[i],mpo[i],B[i],EnvR)
pushright(A::DenseMPS, mpo::SparseMPO, B::T, EnvL::SparseLeftEnvironmentTensor{1}, i::Int64) where T <: Union{AdjointMPS,RefMPS} = pushright(A[i],mpo[i],B[i],EnvL)

function pushleft(A::MPSTensor{3}, mpo::SparseMPOTensor{DL,D,DR}, B::AdjointMPSTensor{3}, EnvR::SparseRightEnvironmentTensor{1}) where {DL,D,DR}
    @assert DR == EnvR.D[1]
    tmpEnvR = Vector{Any}(nothing, D)
    r_map = _validind1(mpo, R2L())
    validind = [(j, r_pairs) for (j, r_pairs) in enumerate(r_map) if !isempty(r_pairs)]
    Nthr = get_nworker()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validind) && break
                j, r_pairs = validind[ct]
                weighted_env = sum(w * EnvR[b] for (b, w) in r_pairs)
                x = contract(A, mpo[j], B, weighted_env)
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
        for (j, r_pairs) in validind
            weighted_env = sum(w * EnvR[b] for (b, w) in r_pairs)
            tmpEnvR[j] = axpy!(1, contract(A, mpo[j], B, weighted_env), tmpEnvR[j])
        end
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor}, tmpEnvR))
end

function pushright(A::MPSTensor{3}, mpo::SparseMPOTensor{DL,D,DR}, B::AdjointMPSTensor{3}, EnvL::SparseLeftEnvironmentTensor{1}) where {DL,D,DR}
    @assert DL == EnvL.D[1]
    tmpEnvL = Vector{Any}(nothing, D)
    l_map = _validind1(mpo, L2R())
    validind = [(j, l_pairs) for (j, l_pairs) in enumerate(l_map) if !isempty(l_pairs)]
    Nthr = get_nworker()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validind) && break
                j, l_pairs = validind[ct]
                weighted_env = sum(w * EnvL[b] for (b, w) in l_pairs)
                x = contract(A, mpo[j], B, weighted_env)
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
        for (j, l_pairs) in validind
            weighted_env = sum(w * EnvL[b] for (b, w) in l_pairs)
            tmpEnvL[j] = axpy!(1, contract(A, mpo[j], B, weighted_env), tmpEnvL[j])
        end
    end
    return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor}, tmpEnvL))
end

pushleft(A::DenseMPO, B::AdjointMPO, EnvR::DenseRightEnvironmentTensor{2}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvR.A))
pushright(A::DenseMPO, B::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{2}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvL.A))

pushleft(A::DenseMPO, B::DenseMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPO, B::DenseMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL.A))

pushleft(A::DenseMPS, B::DenseMPO, C::T₃, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) where T₃ <: Union{AdjointMPS,RefMPS} = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPS, B::DenseMPO, C::T₃, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) where T₃ <: Union{AdjointMPS,RefMPS} = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL.A))

pushleft(A::DenseMPO, B::SparseMPO, C::T, EnvR::SparseRightEnvironmentTensor, i::Int64) where T <: Union{AdjointMPO,RefMPO} = pushleft(A[i],B[i],C[i],EnvR)
pushright(A::DenseMPO, B::SparseMPO, C::T, EnvL::SparseLeftEnvironmentTensor, i::Int64) where T <: Union{AdjointMPO,RefMPO} = pushright(A[i],B[i],C[i],EnvL)
pushleft(A::DenseMPOTensor{4}, B::SparseMPOTensor, C::AdjointMPOTensor{4}, EnvR::SparseRightEnvironmentTensor) = SparseRightEnvironmentTensor(contract(A, B, C, EnvR))
pushright(A::DenseMPOTensor{4}, B::SparseMPOTensor, C::AdjointMPOTensor{4}, EnvL::SparseLeftEnvironmentTensor) = SparseLeftEnvironmentTensor(contract(A, B, C, EnvL))
pushleft(A::DenseMPS, B::AdjointMPS, EnvR::DenseRightEnvironmentTensor{2}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvR.A))
pushright(A::DenseMPS, B::AdjointMPS, EnvL::DenseLeftEnvironmentTensor{2}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B))..., EnvL.A))

#= Env4 =#

function pushright(Hup::SparseMPO, ρ::DenseMPO, Hdown::SparseMPO, ρ′::AdjointMPO, EnvL::SparseLeftEnvironmentTensor{2}, site::Int64)
    tmpEnvL = Array{Any}(nothing, Hup.D[site][2], Hdown.D[site][2])
    l_map_up = _validind1(Hup[site], L2R())
    l_map_down = _validind1(Hdown[site], L2R())
    vind_up = [(j, l_pairs) for (j, l_pairs) in enumerate(l_map_up) if !isempty(l_pairs)]
    vind_down = [(j, l_pairs) for (j, l_pairs) in enumerate(l_map_down) if !isempty(l_pairs)]
    Nthr = get_nworker()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        pairs = [(a,b) for a in eachindex(vind_up) for b in eachindex(vind_down)]
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(pairs) && break
                a, b = pairs[ct]
                op_up, l_pairs_up = vind_up[a]
                op_down, l_pairs_down = vind_down[b]
                for (i, w_i) in l_pairs_up, (j, w_j) in l_pairs_down
                    C = pushright(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], EnvL.A[i,j])
                    lock(Lock)
                    try
                        tmpEnvL[op_up,op_down] = axpy!(w_i * w_j, C, tmpEnvL[op_up,op_down])
                    catch
                        rethrow()
                    finally
                        unlock(Lock)
                    end
                end
            end
        end
    else
        for (op_up, l_pairs_up) in vind_up, (op_down, l_pairs_down) in vind_down
            for (i, w_i) in l_pairs_up, (j, w_j) in l_pairs_down
                C = pushright(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], EnvL.A[i,j])
                tmpEnvL[op_up,op_down] = axpy!(w_i * w_j, C, tmpEnvL[op_up,op_down])
            end
        end
    end
    return SparseLeftEnvironmentTensor(convert(Array{LeftEnvironmentTensor}, tmpEnvL))
end

function pushleft(Hup::SparseMPO, ρ::DenseMPO, Hdown::SparseMPO, ρ′::AdjointMPO, EnvR::SparseRightEnvironmentTensor{2}, site::Int64)
    tmpEnvR = Array{Any}(nothing, Hup.D[site][2], Hdown.D[site][2])
    r_map_up = _validind1(Hup[site], R2L())
    r_map_down = _validind1(Hdown[site], R2L())
    vind_up = [(j, r_pairs) for (j, r_pairs) in enumerate(r_map_up) if !isempty(r_pairs)]
    vind_down = [(j, r_pairs) for (j, r_pairs) in enumerate(r_map_down) if !isempty(r_pairs)]
    Nthr = get_nworker()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        pairs = [(a,b) for a in eachindex(vind_up) for b in eachindex(vind_down)]
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(pairs) && break
                a, b = pairs[ct]
                op_up, r_pairs_up = vind_up[a]
                op_down, r_pairs_down = vind_down[b]
                for (k, w_k) in r_pairs_up, (l, w_l) in r_pairs_down
                    C = pushleft(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], EnvR.A[k,l])
                    lock(Lock)
                    try
                        tmpEnvR[op_up,op_down] = axpy!(w_k * w_l, C, tmpEnvR[op_up,op_down])
                    catch
                        rethrow()
                    finally
                        unlock(Lock)
                    end
                end
            end
        end
    else
        for (op_up, r_pairs_up) in vind_up, (op_down, r_pairs_down) in vind_down
            for (k, w_k) in r_pairs_up, (l, w_l) in r_pairs_down
                C = pushleft(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], EnvR.A[k,l])
                tmpEnvR[op_up,op_down] = axpy!(w_k * w_l, C, tmpEnvR[op_up,op_down])
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
    @timeit to "pushleft" pushleft!(Env)
    @timeit to "back evolve" ~, K = evolve!(tl, projright0(Env;E₀ = info.E), -Alg.τ, Alg.solver)
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
    for (j, r_pairs) in enumerate(r_map)
        isempty(r_pairs) && continue
        weighted_env = sum(w * EnvR[b] for (b, w) in r_pairs)
        tmpEnvR[j] = axpy!(1, contract(A[site][j], B[site], weighted_env), tmpEnvR[j])
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor}, tmpEnvR))
end

function pushright(A::SparseMPO, B::AdjointMPO, EnvL::SparseLeftEnvironmentTensor{1}, site::Int64)
    @assert A.D[site][1] == EnvL.D[1]
    tmpEnvL = Vector{Any}(nothing, A.D[site][2])
    l_map = _validind1(A[site], L2R())
    for (j, l_pairs) in enumerate(l_map)
        isempty(l_pairs) && continue
        weighted_env = sum(w * EnvL[b] for (b, w) in l_pairs)
        tmpEnvL[j] = axpy!(1, contract(A[site][j], B[site], weighted_env), tmpEnvL[j])
    end
    return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor}, tmpEnvL))
end


pushleft(A::DenseMPO, B::AdjointMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvR.A))
pushright(A::DenseMPO, B::AdjointMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(map(x -> x[site],(A,B,C))..., EnvL.A))

pushleft(A::DenseMPO, B::RefMPO, C::AdjointMPO, EnvR::DenseRightEnvironmentTensor{3}, site::Int64) = DenseRightEnvironmentTensor(contract(A[site], B[site], C[site], EnvR.A))
pushright(A::DenseMPO, B::RefMPO, C::AdjointMPO, EnvL::DenseLeftEnvironmentTensor{3}, site::Int64) = DenseLeftEnvironmentTensor(contract(A[site], B[site], C[site], EnvL.A))
