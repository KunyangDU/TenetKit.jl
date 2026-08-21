# ====================== 稠密 bare action（MPS）======================
# DenseProjectiveHamiltonian 的单次作用（每次分配），MPS 侧：MPSTensor{3} / CompositeMPSTensor{2,4}；
# 0-site 纯投影（{3,0}）的 MPSTensor{2}/DenseMPOTensor{2} rank-2 布局一致，在此统一定义。

# ---------- {3,0} 0-site 纯投影（MPSTensor{2} / DenseMPOTensor{2} 共用，rank-2 布局一致）----------

function actionb(O::DenseProjectiveHamiltonian{3,0}, obj::T) where T <: Union{MPSTensor{2}, DenseMPOTensor{2}}
    @tensor x[-1;-2] ≔ O.EnvL.A.A[-1,2,1] * obj.A[1,3] * O.EnvR.A.A[3,2,-2]
    x = T(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

# ---------- {2,1} 两体环境纯投影 ----------

function actionb(O::DenseProjectiveHamiltonian{2,1}, obj::MPSTensor{3})
    @tensor x[-1,-2;-3] ≔ obj.A[1,-2,2] * O.EnvL.A.A[-1,1] * O.EnvR.A.A[2,-3]
    x = MPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

# ---------- {3,1} 三体环境，中间 DenseMPO ----------

function actionb(O::DenseProjectiveHamiltonian{3,1}, obj::MPSTensor{3})
    @tensor x[-1,-2;-3] ≔ O.EnvL.A.A[-1,2,1] * O.H[1].A[-2,2,5,3] * obj.A[1,3,4] * O.EnvR.A.A[4,5,-3]
    x = MPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

# ---------- {3,2} 三体环境，中间 DenseMPO ----------

function actionb(O::DenseProjectiveHamiltonian{3,2}, obj::CompositeMPSTensor{2,4})
    @tensor x[-1,-2,-3;-4] ≔ O.EnvL.A.A[-1,2,1] * O.H[1].A[-2,2,4,3] * O.H[2].A[-3,4,7,5] * obj.A[1,3,5,6] * O.EnvR.A.A[6,7,-4]
    x = CompositeMPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function actionb(O::DenseProjectiveHamiltonian{3,2}, A1::MPSTensor{3}, A2::MPSTensor{3})
    @tensor x[-1,-2,-3;-4] ≔ O.EnvL.A.A[-1,2,1] * O.H[1].A[-2,2,4,3] * O.H[2].A[-3,4,7,5] * A1.A[1,3,8] * A2.A[8,5,6] * O.EnvR.A.A[6,7,-4]
    x = CompositeMPSTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, composite(A1, A2), x))
    return x
end
