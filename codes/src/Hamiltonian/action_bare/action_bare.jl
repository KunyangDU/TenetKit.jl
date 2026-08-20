# ====================== @tensor 裸实现（actionb）======================
# 与零分配 action（action.jl / action_dense.jl）覆盖完全一致，但用 @tensor 形式（每次调用分配中间量）。
# 用途：数值对照（rel err ~1e-16）+ 基准（量化 cache 的加速比）。
# 稀疏路径复用 contract.jl 的 _action*_contract（本就是 @tensor）；
# 稠密路径在此补全 @tensor 表达式（含 MPO / AdjointMPO，此前只有注释）。

# ====================== 稀疏（SparseProjectiveHamiltonian）======================
# 复用 action.jl 的 _action（内部走 _action*_contract 的 @tensor），多线程累加。

# 稀疏累加（多线程）：与 git 改动前 action.jl 相同的模式 —— 每 worker 并行算各 validind 的
# @tensor/contract 贡献 C（昂贵部分无锁），仅对廉价 axpy! 累加加 ReentrantLock；单线程直接串行。
function _sparse_actionb_sum(f::Function, validinds)
    x = nothing
    Nthr = get_nworker()
    n = length(validinds)
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > n && break
                C = f(validinds[ct])
                lock(Lock)
                try
                    x = axpy!(1, C, x)
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for ind in validinds
            x = axpy!(1, f(ind), x)
        end
    end
    return x
end

function actionb(O::SparseProjectiveHamiltonian, obj::T) where T <: Union{MPSTensor{2},DenseMPOTensor{2},MPSTensor{3},DenseMPOTensor{4},CompositeMPSTensor{2,4},CompositeMPOTensor{2,6}}
    x = _sparse_actionb_sum(O.validinds) do ind
        C, _ = _action(O, obj, ind)
        C
    end
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function actionb(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2})
    x = _sparse_actionb_sum(O.validinds) do ind
        C, _ = _action(O, obj, ind)
        C
    end
    return x
end

# ====================== 稠密（DenseProjectiveHamiltonian）======================

# ---------- {2,1} 两体环境纯投影 ----------

function actionb(O::DenseProjectiveHamiltonian{2,1}, obj::DenseMPOTensor{4})
    @tensor x[-1,-2;-3,-4] ≔ O.EnvL.A.A[-2,1] * obj.A[-1,1,2,-4] * O.EnvR.A.A[2,-3]
    x = DenseMPOTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function actionb(O::DenseProjectiveHamiltonian{2,1}, obj::MPSTensor{3})
    @tensor x[-1,-2;-3] ≔ obj.A[1,-2,2] * O.EnvL.A.A[-1,1] * O.EnvR.A.A[2,-3]
    x = MPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

# ---------- {2,2} 两体环境纯投影 ----------

function actionb(O::DenseProjectiveHamiltonian{2,2}, obj::CompositeMPOTensor{2,6})
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ O.EnvL.A.A[-3,1] * obj.A[-1,-2,1,2,-5,-6] * O.EnvR.A.A[2,-4]
    x = CompositeMPOTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

# ---------- {3,1} 三体环境，中间 DenseMPO / AdjointMPO ----------

function actionb(O::DenseProjectiveHamiltonian{3,1}, obj::MPSTensor{3})
    @tensor x[-1,-2;-3] ≔ O.EnvL.A.A[-1,2,1] * O.H[1].A[-2,2,5,3] * obj.A[1,3,4] * O.EnvR.A.A[4,5,-3]
    x = MPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function actionb(O::DenseProjectiveHamiltonian{3,1}, obj::DenseMPOTensor{4})
    x = _actionb1_mpo(O.EnvL.A.A, O.H[1], obj, O.EnvR.A.A)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function _actionb1_mpo(El, h::DenseMPOTensor{4}, obj, Er)
    @tensor x[-1,-2;-3,-4] ≔ El[-2,1,2] * h.A[-1,1,5,3] * obj.A[3,2,4,-4] * Er[4,5,-3]
    return DenseMPOTensor(x)
end

function _actionb1_mpo(El, h::AdjointMPOTensor{4}, obj, Er)
    Erp = permute(Er, ((1,), (2,3)))
    @tensor x[-1,-2;-3,-4] ≔ El[-2,1,2] * h.A[3,-1,4,1] * obj.A[4,2,5,-4] * Erp[5,3,-3]
    return DenseMPOTensor(x)
end

# ---------- {3,2} 三体环境，中间 DenseMPO / AdjointMPO ----------

function actionb(O::DenseProjectiveHamiltonian{3,2}, obj::CompositeMPSTensor{2,4})
    @tensor x[-1,-2,-3;-4] ≔ O.EnvL.A.A[-1,2,1] * O.H[1].A[-2,2,4,3] * O.H[2].A[-3,4,7,5] * obj.A[1,3,5,6] * O.EnvR.A.A[6,7,-4]
    x = CompositeMPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function actionb(O::DenseProjectiveHamiltonian{3,2}, obj::CompositeMPOTensor{2,6})
    x = _actionb2_mpo(O.EnvL.A.A, O.H[1], O.H[2], obj, O.EnvR.A.A)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function _actionb2_mpo(El, h1::DenseMPOTensor{4}, h2::DenseMPOTensor{4}, obj, Er)
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ El[-3,1,2] * h1.A[-2,1,3,4] * h2.A[-1,3,5,6] * obj.A[6,4,2,7,-5,-6] * Er[7,5,-4]
    return CompositeMPOTensor(x)
end

function _actionb2_mpo(El, h1::AdjointMPOTensor{4}, h2::AdjointMPOTensor{4}, obj, Er)
    Erp = permute(Er, ((1,), (2,3)))
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ El[-3,1,2] * h1.A[3,-2,4,1] * h2.A[5,-1,6,3] * obj.A[6,4,2,7,-5,-6] * Erp[7,5,-4]
    return CompositeMPOTensor(x)
end

# ====================== 2-site 分离输入（actionb(O, A1, A2)）======================
# 两个 site 张量分开传入，不再先 composite。缩并逻辑直接内联（不依赖 Algebra/contract.jl）：
# {2,2}/{3,2} MPO 的 @tensor 自 Algebra/contract.jl 逐字搬来；稀疏路径复制其 loop。

function actionb(O::SparseProjectiveHamiltonian{2}, A1::MPSTensor{3}, A2::MPSTensor{3})
    x = _sparse_actionb_sum(O.validinds) do ind
        l_inds, (j, k), r_inds, wl, w_mid, wr = ind
        tmp1 = contract(_wsum(O.EnvL, l_inds, wl), A1, O.H[1][j])
        tmp2 = contract(A2, O.H[2][k], _wsum(O.EnvR, r_inds, wr))
        w_mid * contract(tmp1, tmp2)
    end
    !iszero(O.E₀) && (x = axpy!(-O.E₀, composite(A1, A2), x))
    return x
end

function actionb(O::SparseProjectiveHamiltonian{2}, A1::DenseMPOTensor{4}, A2::DenseMPOTensor{4})
    x = _sparse_actionb_sum(O.validinds) do ind
        l_inds, (j, k), r_inds, wl, w_mid, wr = ind
        tmp1 = contract(_wsum(O.EnvL, l_inds, wl), A1, O.H[1][j])
        tmp2 = contract(A2, O.H[2][k], _wsum(O.EnvR, r_inds, wr))
        w_mid * contract(tmp1, tmp2)
    end
    !iszero(O.E₀) && (x = axpy!(-O.E₀, composite(A1, A2), x))
    return x
end

function actionb(O::DenseProjectiveHamiltonian{2,2}, A1::DenseMPOTensor{4}, A2::DenseMPOTensor{4})
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ O.EnvL.A.A[-3,1] * A1.A[-2,1,2,-6] * A2.A[-1,2,3,-5] * O.EnvR.A.A[3,-4]
    x = CompositeMPOTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, composite(A1, A2), x))
    return x
end

function actionb(O::DenseProjectiveHamiltonian{3,2}, A1::MPSTensor{3}, A2::MPSTensor{3})
    @tensor x[-1,-2,-3;-4] ≔ O.EnvL.A.A[-1,2,1] * O.H[1].A[-2,2,4,3] * O.H[2].A[-3,4,7,5] * A1.A[1,3,8] * A2.A[8,5,6] * O.EnvR.A.A[6,7,-4]
    x = CompositeMPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, composite(A1, A2), x))
    return x
end

function actionb(O::DenseProjectiveHamiltonian{3,2}, A1::DenseMPOTensor{4}, A2::DenseMPOTensor{4})
    x = _actionb2_split(O.EnvL.A.A, A1, A2, O.H[1], O.H[2], O.EnvR.A.A)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, composite(A1, A2), x))
    return x
end

function _actionb2_split(El, A1, A2, h1::DenseMPOTensor{4}, h2::DenseMPOTensor{4}, Er)
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El[-1,1,2] * A1.A[3,2,7,-6] * A2.A[6,7,4,-5] * h1.A[-2,1,8,3] * h2.A[-3,8,5,6] * Er[4,5,-4]
    return CompositeMPOTensor(x)
end

function _actionb2_split(El, A1, A2, h1::AdjointMPOTensor{4}, h2::AdjointMPOTensor{4}, Er)
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El[-1,1,2] * A1.A[3,2,7,-6] * A2.A[6,7,4,-5] * h1.A[8,-2,3,1] * h2.A[5,-3,6,8] * Er[4,5,-4]
    return CompositeMPOTensor(x)
end
