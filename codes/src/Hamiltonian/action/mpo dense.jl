# ====================== 稠密 bare action（MPO）======================
# DenseProjectiveHamiltonian 的单次作用（每次分配），MPO 侧：DenseMPOTensor{4} / CompositeMPOTensor{2,6}，
# 含中间 AdjointMPO 的 _actionb1_mpo / _actionb2_mpo / _actionb2_split。

# ---------- {2,1} 两体环境纯投影 ----------

function actionb(O::DenseProjectiveHamiltonian{2,1}, obj::DenseMPOTensor{4})
    @tensor x[-1,-2;-3,-4] ≔ O.EnvL.A.A[-2,1] * obj.A[-1,1,2,-4] * O.EnvR.A.A[2,-3]
    x = DenseMPOTensor(x)
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

function actionb(O::DenseProjectiveHamiltonian{3,2}, obj::T) where T <: Union{CompositeMPOTensor{2, 6}, AdjointCompositeMPOTensor{2, 6}}
    x = _actionb2_mpo(O.EnvL.A, O.H[1], O.H[2], obj, O.EnvR.A)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))
    return x
end

function _actionb2_mpo(El::LeftEnvironmentTensor{3}, h1::DenseMPOTensor{4}, h2::DenseMPOTensor{4}, obj::CompositeMPOTensor{2,6}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ El.A[-3,1,2] * h1.A[-2,1,3,4] * h2.A[-1,3,5,6] * obj.A[6,4,2,7,-5,-6] * Er.A[7,5,-4]
    return CompositeMPOTensor(x)
end

function _actionb2_mpo(El::LeftEnvironmentTensor{3}, h1::AdjointMPOTensor{4}, h2::AdjointMPOTensor{4}, obj::CompositeMPOTensor{2,6}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ El.A[-3,1,2] * h1.A[3,-2,4,1] * h2.A[5,-1,6,3] * obj.A[6,4,2,7,-5,-6] * Er.A[7,5,-4]
    return CompositeMPOTensor(x)
end

function _actionb2_mpo(El::LeftEnvironmentTensor{3}, h1::DenseMPOTensor{4}, h2::DenseMPOTensor{4}, obj::AdjointCompositeMPOTensor{2, 6}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ El.A[1,2,-6] * h1.A[3,2,4,-5] * h2.A[5,4,7,-4] * obj.A[6,-2,-3,5,3,1] * Er.A[-1,7,6]
    return AdjointCompositeMPOTensor(x)
end

# ---------- 2-site 分离输入（actionb(O, A1, A2)）----------

function actionb(O::DenseProjectiveHamiltonian{2,2}, A1::DenseMPOTensor{4}, A2::DenseMPOTensor{4})
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ O.EnvL.A.A[-3,1] * A1.A[-2,1,2,-6] * A2.A[-1,2,3,-5] * O.EnvR.A.A[3,-4]
    x = CompositeMPOTensor(x)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, composite(A1, A2), x))
    return x
end

function actionb(O::DenseProjectiveHamiltonian{3,2}, A1::T, A2::T) where T <: Union{DenseMPOTensor{4}, AdjointMPOTensor{4}}
    x = _actionb2_split(O.EnvL.A, A1, A2, O.H[1], O.H[2], O.EnvR.A)
    !iszero(O.E₀) && (x = axpy!(-O.E₀, composite(A1, A2), x))
    return x
end

function _actionb2_split(El::LeftEnvironmentTensor{3}, A1::DenseMPOTensor{4}, A2::DenseMPOTensor{4}, h1::DenseMPOTensor{4}, h2::DenseMPOTensor{4}, Er::RightEnvironmentTensor{3})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,1,2] * A1.A[3,2,7,-6] * A2.A[6,7,4,-5] * h1.A[-2,1,8,3] * h2.A[-3,8,5,6] * Er.A[4,5,-4]
    return CompositeMPOTensor(x)
end

function _actionb2_split(El::LeftEnvironmentTensor{3}, A1::DenseMPOTensor{4}, A2::DenseMPOTensor{4}, h1::AdjointMPOTensor{4}, h2::AdjointMPOTensor{4}, Er::RightEnvironmentTensor{3})
    @tensor x[-3,-2,-1;-4,-5,-6] ≔ El.A[-1,1,2] * A1.A[3,2,7,-6] * A2.A[6,7,4,-5] * h1.A[8,-2,3,1] * h2.A[5,-3,6,8] * Er.A[4,5,-4]
    return CompositeMPOTensor(x)
end

function _actionb2_split(El::LeftEnvironmentTensor{3}, A1::AdjointMPOTensor{4}, A2::AdjointMPOTensor{4}, h1::DenseMPOTensor{4}, h2::DenseMPOTensor{4}, Er::RightEnvironmentTensor{3})
    @tensor x[-1,-2,-3;-4,-5,-6] ≔ El.A[1,2,-6] * h1.A[3,2,5,-5] * h2.A[6,5,8,-4] * A1.A[4,-3,3,1] * A2.A[7,-2,6,4] * Er.A[-1,8,7]
    return AdjointCompositeMPOTensor(x)
end

