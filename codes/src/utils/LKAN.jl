# ─────────────────────────────────────────────────────────────────────────────
# Krylov 基底构造
# ─────────────────────────────────────────────────────────────────────────────

"""
    span_basis(obj, O, N)

逐阶构造局域 Krylov 子空间基底。

返回：
  total_G     — AbstractTensorMap 列表，长度 N*L+1，每个 token 为 (d,d;d,d)
  total_basis — MPSTensor 列表，长度 N*L+1，基底张量 {A, EhA_k, Eh(HA)_k, ...}
"""
function span_basis(obj::Union{MPSTensor,DenseMPOTensor}, O::SparseProjectiveHamiltonian, N::Int64; scale::Number = 1.0)
    total_basis = MPSTensor[]
    total_G     = AbstractTensorMap[]

    push!(total_basis, obj)
    push!(total_G, _identity_G(obj))

    # current_level 存储上一阶的所有基向量（初始为 {A}）
    # 每阶对 current_level 中的每个向量单独做 sparse_action，
    # 得到 {H_j b : b ∈ current_level, j ∈ bonds}，即 H_j H_... A。
    current_level = MPSTensor[obj]

    for i in 1:N
        next_level = MPSTensor[]
        for b in current_level
            Gs, Bs = sparse_action(b, O)
            push!(total_G,     Gs...)
            push!(total_basis, Bs...)
            push!(next_level,  Bs...)
        end
        current_level = next_level
    end

    return total_G, total_basis
end

function sparse_action(obj::MPSTensor, O::SparseProjectiveHamiltonian{1})
    L  = length(O.validinds)
    Gs = Vector{AbstractTensorMap}(undef, L)   # token: (d,d;d,d)
    Bs = Vector{MPSTensor}(undef, L)            # basis: (D,d,D)
    for (ind, (i,j)) in enumerate(O.validinds)
        x       = _sparse_action(obj, O.EnvL.A[i], map(x -> x.m[i,j], O.H.ts)..., O.EnvR.A[j])
        Gs[ind] = _sparse_G(x, obj)
        Bs[ind] = _sparse_basis(x)
    end
    return Gs, Bs
end

function _sparse_action(obj::MPSTensor, El::LeftEnvironmentTensor, h::LocalOperator, Er::RightEnvironmentTensor)
    @tensor x[-1,-2,-3;-4,-5] ≔ obj.A[1,-3,2] * El.A[-1,1] * h.A[-2,-4] * Er.A[2,-5]
    return x
end

function _sparse_action(obj::MPSTensor, El::LeftEnvironmentTensor, ::IdentityOperator, Er::RightEnvironmentTensor)
    h = _phy_isometry(obj)
    @tensor x[-1,-2,-3;-4,-5] ≔ obj.A[1,-3,2] * El.A[-1,1] * h[-2,-4] * Er.A[2,-5]
    return x
end

# token G_ij = h_ij ⊗ P_ij，domain=(-1,-2)=(h_in,A_phys)，codomain=(-3,-4)=(h_out,A†_phys)
function _sparse_G(x::AbstractTensorMap{<:Any,3,2}, obj::MPSTensor)
    return @tensor G[-1,-2;-3,-4] ≔ x[1,-1,-2,-3,2] * obj.A'[2,1,-4]
end

# 缩并物理指标：B = EL * A * ER * h，结果为 MPSTensor (D, d_new, D)
function _sparse_basis(x::AbstractTensorMap{<:Any,3,2})
    return MPSTensor(@tensor B[-1,-2;-3] ≔ x[-1,-2,1,1,-3])
end

# M_00 token = I ⊗ ρ，其中 ρ = Tr_virt(A * A†) 为局域约化密度矩阵
function _identity_G(obj::MPSTensor)
    h = _phy_isometry(obj)
    return @tensor G[-1,-2;-3,-4] ≔ h[-1,-3] * obj.A[1,-2,2] * obj.A'[2,1,-4]
end

_phy_isometry(obj::MPSTensor)      = one(scalartype(obj)) * isometry(space(obj)[2], space(obj)[2])
_phy_isometry(obj::DenseMPOTensor) = one(scalartype(obj)) * isometry(space(obj)[1], space(obj)[4])

# ─────────────────────────────────────────────────────────────────────────────
# 辅助：token → Float32 向量（无对称性版本，单 Trivial() block）
#
# G 为 (d,d;d,d)，block(G, Trivial()) = (d²×d²) 矩阵，vec 后得 d⁴ 维向量。
# is_complex=true：拼 [Re; Im]，token_dim = 2*d⁴；否则只取 Re，token_dim = d⁴。
#
# 注意：调用时必须传入全局统一的 is_complex 标志，而非逐 token 自动判断。
# Heisenberg 等模型中 S^y 为纯虚，与 S^z/I 混在同一列表会导致长度不一致。
# ─────────────────────────────────────────────────────────────────────────────
function _token_to_vec(G::AbstractTensorMap; is_complex::Bool = false)
    blk = block(G, Trivial())   # (d², d²) dense matrix
    v   = vec(blk)
    if is_complex
        return Float32.(vcat(real.(v), imag.(v)))   # (2*d⁴,)
    else
        return Float32.(real.(v))                    # (d⁴,)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# lkan_prepare：单格点样本生成（DMRG 和 TDVP 共用 State Head）
#
# 给定当前格点的 MPS 张量 obj 和局域有效哈密顿量 O，构造训练样本 NamedTuple。
#
# 返回字段（对应 LKANArraySample 构造器）：
#   tokens     (token_dim, L+1)    — G tensor 拉平，网络输入（Float32）
#   Gram       (L+1, L+1)          — ⟨B_k|B_l⟩，ComplexF32 Hermitian 矩阵
#   V_new      (L+1,)              — ⟨B_k|A_new⟩，ComplexF32 监督标签
#                                    DMRG：A_new = eigsolve(:SR) 基态（近实数）
#                                    TDVP：A_target（实时演化，含虚部）
#   is_complex Bool                — token 是否含虚部
#
# 参数：
#   obj      — 当前格点 MPS 张量
#   O        — SparseProjectiveHamiltonian（proj1(env, site) 的返回值）
#   N        — Krylov 阶数（k_order）：1, 2 或 3
#   A_target — 监督标签张量（默认 nothing → 自动 eigsolve(:SR)，即 DMRG 模式）
#              传入时直接用于计算 V_new，跳过 eigsolve（TDVP 相位修正后使用）
#   alg      — eigsolve 使用的 Krylov 算法（A_target=nothing 时有效）
# ─────────────────────────────────────────────────────────────────────────────
function lkan_prepare(obj::MPSTensor, O::SparseProjectiveHamiltonian, N::Int;
                      A_target  = nothing,
                      alg       = HamiltonianBoundDefaultLanczos,
                      scale::Number    = 1.,
                      tau_step::Float64 = 0.0)   # 单步实时步长 τ/2（正实数）

    # ── 1. 构造 Krylov 子空间基底 ───────────────────────────────────────────
    total_G, total_basis = span_basis(obj, O, N; scale=scale)
    L1 = length(total_basis)   # L+1 = 基底总数

    # ── 2. Gram 矩阵 G[k,l] = ⟨B_k|B_l⟩（复数 Hermitian）────────────────────
    # v3.5：保留完整复数（不再只取 Re），支持实时 TDVP 的虚部监督信号。
    # Gram 为 ComplexF32 Hermitian 矩阵；对角线（基底模长²）为实数。
    Gram = ComplexF32.([dot(total_basis[k].A, total_basis[l].A)
                        for k in 1:L1, l in 1:L1])
    Heff = ComplexF32.([dot(total_basis[k].A, action(O,total_basis[l]).A)
                        for k in 1:L1, l in 1:L1])

    # ── 3. tokens：将 G tensor 拉平为 Float32 向量 → (token_dim, L+1) ───────
    is_complex = any(g -> eltype(block(g, Trivial())) <: Complex, total_G)
    token_vecs = [_token_to_vec(g; is_complex=is_complex) for g in total_G]
    tokens     = reduce(hcat, token_vecs)   # (token_dim, L+1)

    # ── 4. 监督标签 V_new[k] = ⟨B_k|A_new⟩（复数）──────────────────────────
    # v3.5：保留完整复数。实时 TDVP 中 A_new 包含虚部（虚时方向的动力学），
    # Re 和 Im 均作为监督信号参与损失；DMRG/近实数情况下 Im ≈ 0，自动退化。
    A_new = if isnothing(A_target)
        _, Eg_vecs, _ = eigsolve(x -> action(O, x), obj, 1, :SR, alg.Alg)
        Eg_vecs[1]
    else
        A_target
    end
    V_new = ComplexF32.([dot(total_basis[k].A, A_new.A) for k in 1:L1])

    # ── 5. 返回纯 NamedTuple（张量侧不依赖 LKANTypes）──────────────────────
    # ML 侧用 LKANArraySample(nt.tokens, nt.Gram, nt.V_new; Heff=nt.Heff, is_complex=nt.is_complex)
    # Gram/V_new/Heff 为 ComplexF32，构造器自动处理。
    return (tokens     = tokens,
            Gram       = Gram,
            Heff       = Heff,
            V_new      = V_new,
            tau_step   = tau_step,
            is_complex = is_complex)
end

function lkan_prepare(obj::MPSTensor, O::SparseProjectiveHamiltonian, alg::LKANalgo, tau_step::Number)
    data   = lkan_prepare(obj, O, alg.order; alg=alg.algo, scale=alg.scale, tau_step = abs(tau_step))
    @save "$(alg.filepath)/lkan_data_$(alg.tailname)_$(alg.count).jld2" data
    alg.count += 1
end
