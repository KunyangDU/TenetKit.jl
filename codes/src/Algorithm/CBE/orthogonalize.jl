function orthogonalize!(env::Environment{3},B::Union{DenseMPOTensor{4},MPSTensor{3}},EnvR::SparseRightEnvironmentTensor,osite::Int64)
    EnvRorth = Vector(undef, length(env.layer[2][osite].left.fwd))
    EnvRorth .= nothing

    for (l_inds, j, r_inds, wl, wr) in _validind(env.layer[2][osite])
        tmp = contract(B, env.layer[2][osite][j], _wsum(EnvR, r_inds, wr))
        C = tmp - contract(tmp, B)
        for (i, wi) in zip(l_inds, wl)
            EnvRorth[i] = axpy!(wi, C, EnvRorth[i])
        end
    end

    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
end

function orthogonalize!(env::Environment{3},A::Union{DenseMPOTensor{4},MPSTensor{3}},EnvL::SparseLeftEnvironmentTensor,osite::Int64)
    EnvLorth = Vector(undef, length(env.layer[2][osite].right.rev))
    EnvLorth .= nothing

    for (l_inds, j, r_inds, wl, wr) in _validind(env.layer[2][osite])
        tmp = contract(_wsum(EnvL, l_inds, wl), A, env.layer[2][osite][j])
        C = tmp - contract(tmp, A)
        for (i, wi) in zip(r_inds, wr)
            EnvLorth[i] = axpy!(wi, C, EnvLorth[i])
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
end

function orthogonalize!(H::SparseMPOTensor,B::T,B′::T,EnvR::SparseRightEnvironmentTensor) where T <: Union{DenseMPOTensor{4},MPSTensor{3}}
    EnvRorth = Vector(undef, length(H.left.fwd))
    EnvRorth .= nothing
    validind = _validind(H)
    Nthr = get_nworker()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validind) && break
                l_inds, j, r_inds, wl, wr = validind[ct]
                C = contract(B, H[j], _wsum(EnvR, r_inds, wr)) |> x -> x - contract(x, B′)
                lock(Lock)
                try
                    for (i, wi) in zip(l_inds, wl)
                        EnvRorth[i] = axpy!(wi, C, EnvRorth[i])
                    end
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for (l_inds, j, r_inds, wl, wr) in validind
            C = contract(B, H[j], _wsum(EnvR, r_inds, wr)) |> x -> x - contract(x, B′)
            for (i, wi) in zip(l_inds, wl)
                EnvRorth[i] = axpy!(wi, C, EnvRorth[i])
            end
        end
    end

    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor}, EnvRorth))
end

function orthogonalize!(H::SparseMPOTensor,A::T,A′::T,EnvL::SparseLeftEnvironmentTensor) where T <: Union{DenseMPOTensor{4},MPSTensor{3}}
    EnvLorth = Vector(undef, length(H.right.rev))
    EnvLorth .= nothing
    validind = _validind(H)
    Nthr = get_nworker()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validind) && break
                l_inds, j, r_inds, wl, wr = validind[ct]
                C = contract(_wsum(EnvL, l_inds, wl), A, H[j]) |> x -> x - contract(x, A′)
                lock(Lock)
                try
                    for (i, wi) in zip(r_inds, wr)
                        EnvLorth[i] = axpy!(wi, C, EnvLorth[i])
                    end
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for (l_inds, j, r_inds, wl, wr) in validind
            C = contract(_wsum(EnvL, l_inds, wl), A, H[j]) |> x -> x - contract(x, A′)
            for (i, wi) in zip(r_inds, wr)
                EnvLorth[i] = axpy!(wi, C, EnvLorth[i])
            end
        end
    end

    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor}, EnvLorth))
end

function orthogonalize!(A::Union{DenseMPOTensor{4},MPSTensor{3}},A′::Union{DenseMPOTensor{4},MPSTensor{3}},Env::Union{DenseLeftEnvironmentTensor,DenseRightEnvironmentTensor})
    tmp = contract(Env.A,A)
    Envorth = tmp - contract(tmp,A′)
    return Envorth
end

function orthogonalize!(Q::T,A::T,direction::AbstractDirection;tol::Number=1e-4) where T <: Union{MPSTensor{3},DenseMPOTensor{4},AdjointMPOTensor{4}}
    ϵ = norm(_cbeinner(Q,A,direction))
    ϵ > tol && (ϵ = _cbeorth!(Q,A,direction))
    @assert ϵ < tol ϵ
    return Q
end

orthogonalize!(H::T,A::T′,A′::T′,EnvL::DenseLeftEnvironmentTensor) where {T <: Union{DenseMPOTensor{4},AdjointMPOTensor{4}}, T′ <: Union{DenseMPOTensor{4}, MPSTensor{3}}} = contract(EnvL.A,A,H) |> x -> x - contract(x,A′)
orthogonalize!(H::T,B::T′,B′::T′,EnvR::DenseRightEnvironmentTensor) where {T <: Union{DenseMPOTensor{4},AdjointMPOTensor{4}}, T′ <: Union{DenseMPOTensor{4}, MPSTensor{3}}} = contract(B,H,EnvR.A) |> x -> x - contract(x,B′)

