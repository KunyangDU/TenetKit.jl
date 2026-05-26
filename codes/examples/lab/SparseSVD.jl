using TensorKit, LinearAlgebra

# 块对角矩阵的稀疏SVD
# 输入：A 是 block-diagonal 大矩阵的各个对角块（每块 d×d, d<10）
# 输出：按全局奇异值降序排列的 (λ, u, v, idx) 列表
#   时间复杂度 O(N·d³ + N·d·log(N·d))，内存复用避免重复分配
# 额外要求：全局裁剪，维数D且裁剪掉小于tol的奇异值
function sparse_svd(blocks::Vector{<:AbstractTensorMap}; D::Int=0, tol::Real=0.0)
    # Phase 1: 逐块 SVD，收集 (λ, u_col, v_col, block_idx)
    N = length(blocks)
    # 预分配结果容器：每个块最多 d 个奇异值
    max_len = N * maximum(b -> dim(domain(b)), blocks)  # 上界
    results = Vector{Tuple{Float64,Any,Any,Int}}(undef, max_len)
    pos = 0

    for (k, B) in enumerate(blocks)
        U, S, V = tsvd(B; trunc=notrunc())
        svals = diag(S)[Trivial()]  # 块内奇异值向量
        for i in eachindex(svals)
            λ = svals[i]
            # tol 过滤：跳过近似零的奇异值
            tol > 0 && λ < tol && continue
            pos += 1
            u_i = TensorMap(U.data[:, i], space(U, 1), one(space(U, 2)))
            v_i = TensorMap(V.data[:, i], space(V, 1), one(space(V, 2)))
            results[pos] = (λ, u_i, v_i, k)
        end
    end
    resize!(results, pos)

    # Phase 2: 按奇异值降序排序（用 partialsort 优于 full sort 当仅需 top D）
    if D > 0 && D < length(results)
        # 部分排序：只排 top D，O(K·log(D)) 而非 O(K·log(K))
        partialsort!(results, 1:D; rev=true, by=first)
        resize!(results, D)
    else
        sort!(results; rev=true, by=first)
    end

    return results
end

# 批量版本：返回分离的向量，方便直接使用
function batch_sparse_svd(blocks::Vector{<:AbstractTensorMap}; D::Int=0, tol::Real=0.0)
    res = sparse_svd(blocks; D=D, tol=tol)
    λs = [r[1] for r in res]
    Us  = [r[2] for r in res]
    Vs  = [r[3] for r in res]
    idx = [r[4] for r in res]
    return λs, Us, Vs, idx
end

# ===== 测试 =====
d = 2
N = 100
D = 100
tol = 1e-8

A = [TensorMap(randn, ℂ^d, ℂ^d) for _ in 1:N]

@time result = sparse_svd(A; D=D, tol=tol);
println("Top 5 singular values: ", [r[1] for r in result[1:min(5,end)]])

# 正确性校验：重构块对角矩阵与直接 SVD 比较
# 正确性校验：对少数几个块逐个 tsvd 得到的奇异值，与 sparse_svd 结果一致
A_small = A[1:3]
# 手动收集各块奇异值
svals_manual = Float64[]
for B in A_small
    _, S, _ = tsvd(B; trunc=notrunc())
    append!(svals_manual, diag(S)[Trivial()])
end
sort!(svals_manual; rev=true)
res_full = sparse_svd(A_small)
svals_sparse = [r[1] for r in res_full]
println("Max diff singular values: ", maximum(abs, svals_sparse - svals_manual))




