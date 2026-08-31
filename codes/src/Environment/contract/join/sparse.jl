# join

function contract(EnvL::SparseLeftEnvironmentTensor{1}, EnvR::SparseRightEnvironmentTensor{1}, lm::LayerMap)
    # {0} 直接乘积（无中间算符）：按 source（lm.fwd 的源）遍历，与 proj0/_validind0 一致。
    # 物理上左环境推过 hl 后落在 target（hl 右 bond），右环境推过 hr 后落在 source（hr 左 bond），
    # 故 EnvL 按 target 索引、EnvR 按 source 索引（交叉）。
    @assert EnvL.D[1] == ndst(lm) && EnvR.D[1] == nsrc(lm)
    # 取首个非空项，按 composite 类型直接从空间构造输出零容器（不额外缩并），每 worker 一个
    firstct = nothing
    for a in eachindex(lm.fwd)
        ind = [t[1] for t in lm.fwd[a]]
        isempty(ind) && continue
        firstct = a
        break
    end
    firstct === nothing && return nothing
    El1 = _wsum(EnvL, [t[1] for t in lm.fwd[firstct]], lm.fwd_w[firstct])
    Er1 = EnvR.A[firstct]
    TT = promote_type(scalartype(El1.A), scalartype(Er1.A))
    accs = _join_zerovectors(El1, Er1, TT)
    threaded_reduce!(eachindex(lm.fwd), accs; combine! = (x, y) -> axpy!(1, y, x)) do a, acc, w
        ind = [t[1] for t in lm.fwd[a]]
        isempty(ind) && return acc
        El = _wsum(EnvL, ind, lm.fwd_w[a])
        _accumulate_join!(acc, El, EnvR.A[a])
    end
end

# 函数屏障：按具体 composite 类型构造每 worker 零容器（Vector{T} 具体，累加类型稳定）
_join_zerovectors(El::L, Er::R, TT::Type{<:Number}) where {L, R} = [_join_zero(El, Er, TT) for _ in 1:get_nworker()]

# 按 composite 类型分发，直接从空间生成输出零张量（对标 FiniteMPS 的 zeros(codomain, domain)）。
# 输出空间与下面 contract 的 @tensor 指标模式一一对应（含 codomain 反转 / 带出 domain 腿）。
# —— 左环境在前（El=Left，Er=Right），与顶层 contract(SparseLeft, SparseRight) 语义一致
_join_zero(El::LeftCompositeEnvironmentTensor{2,3}, Er::RightEnvironmentTensor{2}, TT::Type{<:Number}) =
    MPSTensor(zeros(TT, codomain(El.A), domain(Er.A)))
_join_zero(El::LeftCompositeEnvironmentTensor{2,4}, Er::RightEnvironmentTensor{3}, TT::Type{<:Number}) =
    MPSTensor(zeros(TT, codomain(El.A), domain(Er.A)))
_join_zero(El::LeftCompositeEnvironmentTensor{2,4}, Er::RightEnvironmentTensor{2}, TT::Type{<:Number}) =
    DenseMPOTensor(zeros(TT, reverse(codomain(El.A)), domain(Er.A) ⊗ domain(El.A)[2]))
_join_zero(El::LeftCompositeEnvironmentTensor{2,5}, Er::RightEnvironmentTensor{3}, TT::Type{<:Number}) =
    DenseMPOTensor(zeros(TT, reverse(codomain(El.A)), domain(Er.A) ⊗ domain(El.A)[3]))
_join_zero(El::LeftEnvironmentTensor{3}, Er::RightCompositeEnvironmentTensor{1,4}, TT::Type{<:Number}) =
    MPSTensor(zeros(TT, codomain(El.A) ⊗ codomain(Er.A)[3], domain(Er.A)))
_join_zero(El::LeftEnvironmentTensor{2}, Er::RightCompositeEnvironmentTensor{1,3}, TT::Type{<:Number}) =
    MPSTensor(zeros(TT, codomain(El.A) ⊗ codomain(Er.A)[2], domain(Er.A)))
_join_zero(El::LeftEnvironmentTensor{3}, Er::RightCompositeEnvironmentTensor{2,5}, TT::Type{<:Number}) =
    DenseMPOTensor(zeros(TT, codomain(Er.A)[3] ⊗ codomain(El.A), domain(Er.A)))
_join_zero(El::LeftEnvironmentTensor{2}, Er::RightCompositeEnvironmentTensor{2,4}, TT::Type{<:Number}) =
    DenseMPOTensor(zeros(TT, codomain(Er.A)[2] ⊗ codomain(El.A), domain(Er.A)))

# 就地融合累加：@tensor acc.A += El.A*Er.A 对任意收缩（含多索引）成立，等价 mul!(acc,El,Er,1,1)。
# 每 worker 预分配的零容器作为累加目标，全项原地累加，不物化 contract(El,Er) 中间量。
# —— 左环境在前（El=Left，Er=Right），与顶层 contract(SparseLeft, SparseRight) 语义一致
function _accumulate_join!(acc::MPSTensor{3}, El::LeftCompositeEnvironmentTensor{2,3}, Er::RightEnvironmentTensor{2})
    @tensor acc.A[-1,-2;-3] += El.A[-1,-2,1] * Er.A[1,-3]
    return acc
end
function _accumulate_join!(acc::MPSTensor{3}, El::LeftCompositeEnvironmentTensor{2,4}, Er::RightEnvironmentTensor{3})
    @tensor acc.A[-1,-2;-3] += El.A[-1,-2,2,1] * Er.A[1,2,-3]
    return acc
end
function _accumulate_join!(acc::DenseMPOTensor{4}, El::LeftCompositeEnvironmentTensor{2,4}, Er::RightEnvironmentTensor{2})
    @tensor acc.A[-1,-2;-3,-4] += El.A[-2,-1,1,-4] * Er.A[1,-3]
    return acc
end
function _accumulate_join!(acc::DenseMPOTensor{4}, El::LeftCompositeEnvironmentTensor{2,5}, Er::RightEnvironmentTensor{3})
    @tensor acc.A[-1,-2;-3,-4] += El.A[-2,-1,2,1,-4] * Er.A[1,2,-3]
    return acc
end
function _accumulate_join!(acc::MPSTensor{3}, El::LeftEnvironmentTensor{3}, Er::RightCompositeEnvironmentTensor{1,4})
    @tensor acc.A[-1,-2;-3] += El.A[-1,2,1] * Er.A[1,2,-2,-3]
    return acc
end
function _accumulate_join!(acc::MPSTensor{3}, El::LeftEnvironmentTensor{2}, Er::RightCompositeEnvironmentTensor{1,3})
    @tensor acc.A[-1,-2;-3] += El.A[-1,1] * Er.A[1,-2,-3]
    return acc
end
function _accumulate_join!(acc::DenseMPOTensor{4}, El::LeftEnvironmentTensor{3}, Er::RightCompositeEnvironmentTensor{2,5})
    @tensor acc.A[-1,-2;-3,-4] += El.A[-2,2,1] * Er.A[1,2,-1,-3,-4]
    return acc
end
function _accumulate_join!(acc::DenseMPOTensor{4}, El::LeftEnvironmentTensor{2}, Er::RightCompositeEnvironmentTensor{2,4})
    @tensor acc.A[-1,-2;-3,-4] += El.A[-2,1] * Er.A[1,-1,-3,-4]
    return acc
end
_accumulate_join!(acc, El, Er) = axpy!(1, contract(El, Er), acc)
