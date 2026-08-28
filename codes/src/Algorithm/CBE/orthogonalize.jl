# function orthogonalize!(env::Environment{3},B::Union{DenseMPOTensor{4},MPSTensor{3}},EnvR::SparseRightEnvironmentTensor,osite::Int64)
#     EnvRorth = Vector(undef, length(env.layer[2][osite].left.fwd))
#     EnvRorth .= nothing
#     validind = _validind(env.layer[2][osite])
#     n = length(validind)
#     # 逆索引 i -> [(k, wi)]：输出槽 i 被哪些 validind 项（k）以权重 wi 散射（左键 l_inds）
#     inverse = [Any[] for _ in eachindex(EnvRorth)]
#     for k in 1:n
#         l_inds, j, r_inds, wl, wr = validind[k]
#         for (idx, i) in enumerate(l_inds)
#             push!(inverse[i], (k, wl[idx]))
#         end
#     end
#     # Phase 1：并行算每个 validind 项的 C（不相交写 tmpC[k]）
#     tmpC = Vector{Any}(nothing, n)
#     threaded_foreach(eachindex(validind)) do k
#         l_inds, j, r_inds, wl, wr = validind[k]
#         tmp = contract(B, env.layer[2][osite][j], _wsum(EnvR, r_inds, wr))
#         tmpC[k] = _orth_sub!(tmp, B)
#     end
#     # Phase 2：并行按输出槽散射（不相交写 EnvRorth[i]）
#     threaded_foreach(eachindex(EnvRorth)) do i
#         acc = nothing
#         for (k, wi) in inverse[i]
#             acc = axpy!(wi, tmpC[k], acc)
#         end
#         EnvRorth[i] = acc
#     end
#     return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor},EnvRorth))
# end

# function orthogonalize!(env::Environment{3},A::Union{DenseMPOTensor{4},MPSTensor{3}},EnvL::SparseLeftEnvironmentTensor,osite::Int64)
#     EnvLorth = Vector(undef, length(env.layer[2][osite].right.rev))
#     EnvLorth .= nothing
#     validind = _validind(env.layer[2][osite])
#     n = length(validind)
#     # 逆索引 i -> [(k, wi)]：输出槽 i 被哪些 validind 项（k）以权重 wi 散射（右键 r_inds）
#     inverse = [Any[] for _ in eachindex(EnvLorth)]
#     for k in 1:n
#         l_inds, j, r_inds, wl, wr = validind[k]
#         for (idx, i) in enumerate(r_inds)
#             push!(inverse[i], (k, wr[idx]))
#         end
#     end
#     # Phase 1：并行算每个 validind 项的 C（不相交写 tmpC[k]）
#     tmpC = Vector{Any}(nothing, n)
#     threaded_foreach(eachindex(validind)) do k
#         l_inds, j, r_inds, wl, wr = validind[k]
#         tmp = contract(_wsum(EnvL, l_inds, wl), A, env.layer[2][osite][j])
#         tmpC[k] = _orth_sub!(tmp, A)
#     end
#     # Phase 2：并行按输出槽散射（不相交写 EnvLorth[i]）
#     threaded_foreach(eachindex(EnvLorth)) do i
#         acc = nothing
#         for (k, wi) in inverse[i]
#             acc = axpy!(wi, tmpC[k], acc)
#         end
#         EnvLorth[i] = acc
#     end
#     return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor},EnvLorth))
# end

# 逐元素归并 per-worker 私有输出向量（键索引 → 累加张量），供 threaded_reduce! 的 combine! 使用
function _merge_envs!(x::Vector{Any}, y::Vector{Any})
    for i in eachindex(x)
        if x[i] === nothing
            x[i] = y[i]
        elseif y[i] !== nothing
            axpy!(1, y[i], x[i])
        end
    end
    return x
end

function orthogonalize!(H::SparseMPOTensor,B::T,B′::T,EnvR::SparseRightEnvironmentTensor) where T <: Union{DenseMPOTensor{4},MPSTensor{3}}
    # 算子优先（元任务 = _validind 的每个算符 j）：预求和右环境 → 一次缩并 → 按左键散射。
    # 旧 bond-first 版本同一算符的每个左键都重复缩并一次；这里每算符只缩并一次，任务数 = 算符数（≫ 键数）。
    # 散射用 per-worker 私有输出向量 + 末步归并，无嵌套循环、无 barrier、不物化整份 tmpC。
    validind = _validind(H)
    accs = [Vector{Any}(nothing, length(H.left.fwd)) for _ in 1:get_nworker()]
    merged = threaded_reduce!(eachindex(validind), accs; combine! = _merge_envs!) do k, acc, _
        l_inds, j, r_inds, wl, wr = validind[k]
        weighted_env = _wsum(EnvR, r_inds, wr)
        C = _orth_sub!(contract(B, H[j], weighted_env), B′)
        for (idx, i) in enumerate(l_inds)
            acc[i] = axpy!(wl[idx], C, acc[i])
        end
        acc
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightCompositeEnvironmentTensor}, merged))
end

function orthogonalize!(H::SparseMPOTensor,A::T,A′::T,EnvL::SparseLeftEnvironmentTensor) where T <: Union{DenseMPOTensor{4},MPSTensor{3}}
    # 算子优先（pushright 镜像）：预求和左环境 → 一次缩并 → 按右键散射，同样每算符只缩并一次。
    validind = _validind(H)
    accs = [Vector{Any}(nothing, length(H.right.rev)) for _ in 1:get_nworker()]
    merged = threaded_reduce!(eachindex(validind), accs; combine! = _merge_envs!) do k, acc, _
        l_inds, j, r_inds, wl, wr = validind[k]
        weighted_env = _wsum(EnvL, l_inds, wl)
        C = _orth_sub!(contract(weighted_env, A, H[j]), A′)
        for (idx, i) in enumerate(r_inds)
            acc[i] = axpy!(wr[idx], C, acc[i])
        end
        acc
    end
    return SparseLeftEnvironmentTensor(convert(Vector{LeftCompositeEnvironmentTensor}, merged))
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

