using TensorKit, LinearAlgebra, JLD2

include("../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/Heisenberg/trivial/data"

global HamiltonianBoundDefaultLanczos = Krylovalgo(KrylovKit.Lanczos(krylovdim=16, maxiter=2, tol=1e-4, orth=ModifiedGramSchmidt(), eager=true, verbosity=0))

# ─────────────────────────────────────────────────────────────────────────────
# Krylov 基底构造
# ─────────────────────────────────────────────────────────────────────────────

"""
    span_basis(obj, O, N)

逐阶构造局域 Krylov 子空间基底。

返回：
  total_G           — AbstractTensorMap 列表，长度 N*L+1，每个 token 为 (d,d;d,d)
  total_basis       — MPSTensor 列表，长度 N*L+1，基底张量 {A, EhA_k, Eh(HA)_k, ...}
  total_basis_coord — Vector{Float32} 列表，长度 N+1，各命名向量在基底上的坐标
                        coord[1] = [1,0,...,0]        (:A)
                        coord[2] = [0,1,...,1,0,...,0] (:HA)
                        coord[3] = [0,...,0,1,...,1]   (:H2A，N=2 时)
"""
function span_basis(obj::Union{MPSTensor,DenseMPOTensor}, O::SparseProjectiveHamiltonian, N::Int64)
    total_basis = MPSTensor[]
    total_G = AbstractTensorMap[]
    total_basis_coord = Vector[]

    push!(total_basis, obj)
    push!(total_basis_coord, [1.0,])
    push!(total_G, _identity_G(obj))

    for _ in 1:N
        Gs, Bs = sparse_action(obj, O)
        obj = sum(Bs)
        push!(total_basis, Bs...)
        push!(total_basis_coord, repeat([1.0,], length(Bs)))
        push!(total_G, Gs...)
    end

    total_basis_coord = map(x -> vcat(x...),
        eachrow([t for _ in eachindex(total_basis_coord), t in total_basis_coord] .*
                diagm(ones(length(total_basis_coord)))))
    return total_G, total_basis, total_basis_coord
end

function sparse_action(obj::MPSTensor, O::SparseProjectiveHamiltonian{1})
    L = length(O.validinds)
    Gs = Vector{AbstractTensorMap}(undef, L)   # token: (d,d;d,d)
    Bs = Vector{MPSTensor}(undef, L)            # basis: (D,d,D)
    for (ind, (i,j)) in enumerate(O.validinds)
        x = _sparse_action(obj, O.EnvL.A[i], map(x -> x.m[i,j], O.H.ts)..., O.EnvR.A[j])
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

_phy_isometry(obj::MPSTensor)      = isometry(space(obj)[2], space(obj)[2])
_phy_isometry(obj::DenseMPOTensor) = isometry(space(obj)[1], space(obj)[4])

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
# lkan_prepare：单格点样本生成
#
# 给定当前格点的 MPS 张量 obj 和局域有效哈密顿量 O，构造 NamedTuple 样本。
# 通过 mode 控制 eigsolve 开销，只计算训练所需的监督信号：
#
#   mode=:dmrg  — 仅 eigsolve(:SR)，返回 V_new（State Head 监督信号）
#                  E_spec_lo/hi = NaN32
#   mode=:tdvp  — eigsolve(:SR) + eigsolve(:LR)，返回 E_spec_lo/hi（Energy Head）
#                  V_new = Float32[]
#   mode=:both  — 同上两者，全字段填充（适合一次采集两种训练数据）
#
# 参数：
#   obj  — 当前格点 MPS 张量
#   O    — SparseProjectiveHamiltonian（proj1(env, site) 的返回值）
#   N    — Krylov 阶数（k_order）：1, 2 或 3
#   mode — :dmrg / :tdvp / :both（默认 :dmrg）
#   alg  — eigsolve 使用的 Krylov 算法
# ─────────────────────────────────────────────────────────────────────────────
function lkan_prepare(obj::MPSTensor, O::SparseProjectiveHamiltonian, N::Int;
                      mode::Symbol = :dmrg,
                      alg          = HamiltonianBoundDefaultLanczos)
    @assert mode in (:dmrg, :tdvp, :both) "mode 须为 :dmrg、:tdvp 或 :both"

    # ── 1. 构造 Krylov 子空间基底 ───────────────────────────────────────────
    total_G, total_basis, total_basis_coord = span_basis(obj, O, N)
    L1 = length(total_basis)   # L+1 = 基底总数

    # ── 2. Gram 矩阵 G[k,l] = Re(⟨B_k|B_l⟩) ────────────────────────────────
    # 对 Hermitian H，Gram 的虚部反对称，实系数 loss 中自动消除，只存实部
    Gram = Float32.([real(dot(total_basis[k].A, total_basis[l].A))
                     for k in 1:L1, l in 1:L1])

    # ── 3. tokens：将 G tensor 拉平为 Float32 向量 → (token_dim, L+1) ───────
    # 先扫描全部 G 判断是否有复数 block，再统一格式，避免混批时向量长度不一致
    is_complex = any(g -> eltype(block(g, Trivial())) <: Complex, total_G)
    token_vecs = [_token_to_vec(g; is_complex=is_complex) for g in total_G]
    tokens     = reduce(hcat, token_vecs)   # (token_dim, L+1)

    # ── 4. coords Dict：:A, :HA, :H2A, :H3A ────────────────────────────────
    coord_keys = [:A, :HA, :H2A, :H3A]
    coords = Dict{Symbol, Vector{Float32}}(
        coord_keys[i] => Float32.(total_basis_coord[i]) for i in 1:N+1
    )

    # ── 5. eigsolve（按 mode 选择）──────────────────────────────────────────
    V_new     = Float32[]
    E_spec_lo = NaN32
    E_spec_hi = NaN32

    if mode == :dmrg || mode == :both
        # :SR → 基态本征向量，用于 State Head 监督信号 V_new
        Eg_vals, Eg_vecs, _ = eigsolve(x -> action(O, x), obj, 1, :SR, alg.Alg)
        A_new  = Eg_vecs[1]
        V_new  = Float32.([real(dot(total_basis[k].A, A_new.A)) for k in 1:L1])
        if mode == :both
            E_spec_lo = Float32(real(Eg_vals[1]))   # :both 时顺带记录下界
        end
    end

    if mode == :tdvp || mode == :both
        # :SR → 能谱下界；:LR → 能谱上界
        if mode == :tdvp   # :both 的 :SR 已在上面运行，避免重复
            Eg_vals, _, _ = eigsolve(x -> action(O, x), obj, 1, :SR, alg.Alg)
            E_spec_lo = Float32(real(Eg_vals[1]))
        end
        Eh_vals, _, _ = eigsolve(x -> action(O, x), obj, 1, :LR, alg.Alg)
        E_spec_hi = Float32(real(Eh_vals[1]))
    end

    # ── 6. 返回纯 NamedTuple（张量侧不依赖 LKANTypes）──────────────────────
    # ML 侧用 LKANArraySample(nt.tokens, nt.Gram, nt.coords, nt.V_new,
    #                          nt.E_spec_lo, nt.E_spec_hi; is_complex=nt.is_complex)
    return (tokens     = tokens,
            Gram       = Gram,
            coords     = coords,
            V_new      = V_new,
            E_spec_lo  = E_spec_lo,
            E_spec_hi  = E_spec_hi,
            is_complex = is_complex)
end

function lkan_prepare(obj::MPSTensor, O::SparseProjectiveHamiltonian, alg::LKANalgo)
    data = lkan_prepare(obj, O, alg.N; mode = alg.mode, alg = alg.algo)
    @save "$(alg.filepath)/lkan_data_$(alg.tailname).jld2" data
end



