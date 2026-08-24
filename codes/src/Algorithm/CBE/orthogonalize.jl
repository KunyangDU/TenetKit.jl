function orthogonalize!(env::Environment{3},B::Union{DenseMPOTensor{4},MPSTensor{3}},EnvR::SparseRightEnvironmentTensor,osite::Int64)
    EnvRorth = Vector(undef, length(env.layer[2][osite].left.fwd))
    EnvRorth .= nothing
    validind = _validind(env.layer[2][osite])
    n = length(validind)
    # 逆索引 i -> [(k, wi)]：输出槽 i 被哪些 validind 项（k）以权重 wi 散射（左键 l_inds）
    inverse = [Any[] for _ in eachindex(EnvRorth)]
    for k in 1:n
        l_inds, j, r_inds, wl, wr = validind[k]
        for (idx, i) in enumerate(l_inds)
            push!(inverse[i], (k, wl[idx]))
        end
    end
    # Phase 1：并行算每个 validind 项的 C（不相交写 tmpC[k]）
    tmpC = Vector{Any}(nothing, n)
    threaded_foreach(eachindex(validind)) do k
        l_inds, j, r_inds, wl, wr = validind[k]
        tmp = contract(B, env.layer[2][osite][j], _wsum(EnvR, r_inds, wr))
        tmpC[k] = tmp - contract(tmp, B)
    end
    # Phase 2：并行按输出槽散射（不相交写 EnvRorth[i]）
    threaded_foreach(eachindex(EnvRorth)) do i
        acc = nothing
        for (k, wi) in inverse[i]
            acc = axpy!(wi, tmpC[k], acc)
        end
        EnvRorth[i] = acc
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
end

function orthogonalize!(env::Environment{3},A::Union{DenseMPOTensor{4},MPSTensor{3}},EnvL::SparseLeftEnvironmentTensor,osite::Int64)
    EnvLorth = Vector(undef, length(env.layer[2][osite].right.rev))
    EnvLorth .= nothing
    validind = _validind(env.layer[2][osite])
    n = length(validind)
    # 逆索引 i -> [(k, wi)]：输出槽 i 被哪些 validind 项（k）以权重 wi 散射（右键 r_inds）
    inverse = [Any[] for _ in eachindex(EnvLorth)]
    for k in 1:n
        l_inds, j, r_inds, wl, wr = validind[k]
        for (idx, i) in enumerate(r_inds)
            push!(inverse[i], (k, wr[idx]))
        end
    end
    # Phase 1：并行算每个 validind 项的 C（不相交写 tmpC[k]）
    tmpC = Vector{Any}(nothing, n)
    threaded_foreach(eachindex(validind)) do k
        l_inds, j, r_inds, wl, wr = validind[k]
        tmp = contract(_wsum(EnvL, l_inds, wl), A, env.layer[2][osite][j])
        tmpC[k] = tmp - contract(tmp, A)
    end
    # Phase 2：并行按输出槽散射（不相交写 EnvLorth[i]）
    threaded_foreach(eachindex(EnvLorth)) do i
        acc = nothing
        for (k, wi) in inverse[i]
            acc = axpy!(wi, tmpC[k], acc)
        end
        EnvLorth[i] = acc
    end
    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
end

function orthogonalize!(H::SparseMPOTensor,B::T,B′::T,EnvR::SparseRightEnvironmentTensor) where T <: Union{DenseMPOTensor{4},MPSTensor{3}}
    EnvRorth = Vector(undef, length(H.left.fwd))
    EnvRorth .= nothing
    validind = _validind(H)
    n = length(validind)
    # 逆索引 i -> [(k, wi)]：输出槽 i 被哪些 validind 项（k）以权重 wi 散射（左键 l_inds）
    inverse = [Any[] for _ in eachindex(EnvRorth)]
    for k in 1:n
        l_inds, j, r_inds, wl, wr = validind[k]
        for (idx, i) in enumerate(l_inds)
            push!(inverse[i], (k, wl[idx]))
        end
    end
    # Phase 1：并行算每个 validind 项的 C（不相交写 tmpC[k]）
    tmpC = Vector{Any}(nothing, n)
    threaded_foreach(eachindex(validind)) do k
        l_inds, j, r_inds, wl, wr = validind[k]
        tmpC[k] = contract(B, H[j], _wsum(EnvR, r_inds, wr)) |> x -> x - contract(x, B′)
    end
    # Phase 2：并行按输出槽散射（不相交写 EnvRorth[i]）
    threaded_foreach(eachindex(EnvRorth)) do i
        acc = nothing
        for (k, wi) in inverse[i]
            acc = axpy!(wi, tmpC[k], acc)
        end
        EnvRorth[i] = acc
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor}, EnvRorth))
end

function orthogonalize!(H::SparseMPOTensor,A::T,A′::T,EnvL::SparseLeftEnvironmentTensor) where T <: Union{DenseMPOTensor{4},MPSTensor{3}}
    EnvLorth = Vector(undef, length(H.right.rev))
    EnvLorth .= nothing
    validind = _validind(H)
    n = length(validind)
    # 逆索引 i -> [(k, wi)]：输出槽 i 被哪些 validind 项（k）以权重 wi 散射（右键 r_inds）
    inverse = [Any[] for _ in eachindex(EnvLorth)]
    for k in 1:n
        l_inds, j, r_inds, wl, wr = validind[k]
        for (idx, i) in enumerate(r_inds)
            push!(inverse[i], (k, wr[idx]))
        end
    end
    # Phase 1：并行算每个 validind 项的 C（不相交写 tmpC[k]）
    tmpC = Vector{Any}(nothing, n)
    threaded_foreach(eachindex(validind)) do k
        l_inds, j, r_inds, wl, wr = validind[k]
        tmpC[k] = contract(_wsum(EnvL, l_inds, wl), A, H[j]) |> x -> x - contract(x, A′)
    end
    # Phase 2：并行按输出槽散射（不相交写 EnvLorth[i]）
    threaded_foreach(eachindex(EnvLorth)) do i
        acc = nothing
        for (k, wi) in inverse[i]
            acc = axpy!(wi, tmpC[k], acc)
        end
        EnvLorth[i] = acc
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

