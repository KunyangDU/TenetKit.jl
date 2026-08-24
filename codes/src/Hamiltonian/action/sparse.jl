# ====================== 稀疏 bare action（actionb）======================
# 单次作用、每次分配。稀疏路径复用 contract.jl 的 _action*_contract（本就是 @tensor），
# 多线程累加。入口 actionb(Sparse, obj) 用 Union 统一方法，靠 obj 类型派发到 _action*。

# 稀疏累加（多线程）：per-thread 私有累加器 + 末步确定性归约（threaded_reduce!，无锁）。
function _sparse_actionb_sum(f::Function, validinds)
    accs = Vector{Any}(nothing, get_nworker())
    threaded_reduce!(validinds, accs; combine! = (x, y) -> axpy!(1, y, x)) do ind, acc, w
        axpy!(1, f(ind), acc)
    end
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

# ====================== 稀疏 bare 派发（_action）======================
# _action 按 N（0/1/2）与 obj 类型派发，内部走 contract.jl 的 _action*_contract（@tensor）。

function _action(O::SparseProjectiveHamiltonian{0}, obj::T, ind::Tuple{Vector{Int64},Nothing,Vector{Int64},Vector{Number},Vector{Number}}) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    l_inds, ~, r_inds, wl, wr = ind
    tmp,localto = _action0(obj, _wsum(O.EnvL, l_inds, wl), _wsum(O.EnvR, r_inds, wr))
    return tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{1}, obj::T, ind::Tuple{Vector{Int64},Int64,Vector{Int64},Vector{Number},Vector{Number}}) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    l_inds, j, r_inds, wl, wr = ind
    tmp,localto = _action1(obj, _wsum(O.EnvL, l_inds, wl), O.H[1][j], _wsum(O.EnvR, r_inds, wr))
    return tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{2}, obj::T, ind::Tuple{Vector{Int64},Tuple{Int64,Int64},Vector{Int64},Vector{Number},Number,Vector{Number}}) where T <: Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}}
    l_inds, (j,k), r_inds, wl, w_mid, wr = ind
    tmp,localto = _action2(obj, _wsum(O.EnvL, l_inds, wl), O.H[1][j], O.H[2][k], _wsum(O.EnvR, r_inds, wr))
    return w_mid * tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2}, ind::Tuple{Vector{Int64},Tuple{Int64,Int64},Vector{Int64},Vector{Number},Number,Vector{Number}})
    l_inds, (j,k), r_inds, wl, w_mid, wr = ind
    localto = TimerOutput()
    @timeit localto "_action2_EL1=El_H1" EL1 = contract(_wsum(O.EnvL, l_inds, wl), obj[1][j])
    @timeit localto "_action2_EL2=EL1_H2" EL2 = contract(EL1, obj[2][k])
    @timeit localto "_action2_C=EL2_Er" C = contract(EL2, _wsum(O.EnvR, r_inds, wr))
    return w_mid * C, localto
end

function _action0(obj::T,El::LeftEnvironmentTensor{el},Er::RightEnvironmentTensor{er}) where {el,er, T <: Union{MPSTensor{2},DenseMPOTensor{2}}}
    localto = TimerOutput()
    @timeit localto "_action0_0_$(el)_$(er)" tmp = _action0_contract(obj,El,Er)
    return T(tmp),localto
end

function _action1(obj::T,El::LeftEnvironmentTensor{el},h::AbstractLocalOperator{h1,h2},Er::RightEnvironmentTensor{er}) where {el,h1,h2,er, T <: Union{DenseMPOTensor{4},MPSTensor{3}}}
    localto = TimerOutput()
    @timeit localto "_action1_1_$(el)_$(h1)$(h2)_$(er)" tmp = _action1_contract(obj,El,h,Er)
    return T(tmp), localto
end

function _action2(obj::T,El::LeftEnvironmentTensor{el},hl::AbstractLocalOperator{hl1,hl2},hr::AbstractLocalOperator{hr1,hr2},Er::RightEnvironmentTensor{er}) where {el,hl1,hl2,hr1,hr2,er, T<:Union{CompositeMPOTensor{2,6},CompositeMPSTensor{2,4}}}
    localto = TimerOutput()
    @timeit localto "_action2_2_$(el)_$(hl1)$(hl2)_$(hr1)$(hr2)_$(er)" tmp = _action2_contract(obj,El,hl,hr,Er)
    return T(tmp), localto
end

# ====================== 2-site 分离输入（actionb(O, A1, A2)）======================
# 两个 site 张量分开传入，不再先 composite。缩并逻辑直接内联（不依赖 Algebra/contract.jl）。

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
