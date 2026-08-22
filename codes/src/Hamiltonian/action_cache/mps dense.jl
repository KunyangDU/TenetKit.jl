# ====================== 稠密 cache action（MPS）======================
# DenseProjectiveHamiltonian 的零分配作用，MPS 侧：MPSTensor{3} / CompositeMPSTensor{2,4}。
# 零分配约定：稠密路径无 validinds 循环（单次缩并），缓存单套中间缓冲 + 单个累加器，单线程无 @spawn。
# 末步 add_permute!(acc.A, cache[end], perm, 1, 1) 融合「permute + 累加」，acc 已被 zerovector! 清零。

# ====================== {3,0}：0-site 纯投影（无算符，零分配）======================
# 纯投影网络 El·obj·Er（rank-3 环境夹 rank-2 bond 张量）。旧 @tensor 参考：
#   x[-1;-2] ≔ EnvL[-1,2,1] * obj[1,3] * EnvR[3,2,-2]   → [α; δ]

function action(O::DenseProjectiveHamiltonian{3,0}, obj::T) where T <: Union{MPSTensor{2}, DenseMPOTensor{2}}
    c = O.cache === nothing ? _init_proj_action_cache!(O, obj) : O.cache
    to = get_timer("action")
    objA = obj.A * one(c.TT)
    objW = T(objA)
    TensorKit.zerovector!(c.acc)
    @timeit to "_proj_action0" _proj_action0_contract!(c.cache, c.acc, objW, O.EnvL, O.EnvR)
    x = c.acc::T
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    return x
end

# 0-site 纯投影：El[α;σ,γ] · obj[γ;β] · Er[β,σ;δ] → [α;δ]（与稀疏 {3}/{3} 同构，仅包装不同）
function _proj_action0_contract!(cache::Vector{Any}, acc::T, objW::T,
        ElW::DenseLeftEnvironmentTensor{3}, ErW::DenseRightEnvironmentTensor{3}) where T <: Union{MPSTensor{2}, DenseMPOTensor{2}}
    objA = objW.A; El = ElW.A.A; Er = ErW.A.A
    if isempty(cache)
        c1 = permute(Er, ((1,), (2,3)); copy=true)    # [β; σ, δ]
        c2 = objA * c1                                 # [γ; σ, δ]
        c3 = permute(c2, ((2,1), (3,)); copy=true)     # [σ, γ; δ]
        empty!(cache); append!(cache, [c1, c2, c3])
    else
        permute!(cache[1], Er, ((1,), (2,3)))
        TensorKit.mul!(cache[2], objA, cache[1], 1, 0)
        permute!(cache[3], cache[2], ((2,1), (3,)))
    end
    TensorKit.mul!(acc.A, El, cache[3], 1, 1)     # acc += [α; δ]
    return acc
end

# ====================== {2,1}：两体环境纯投影（无算符，零分配）======================
# 纯投影网络 El·obj·Er（phys 透传）。旧 @tensor 参考：
#   {2,1} MPS: tmp[-1,-2;-3] ≔ A[1,-2,2] * El[-1,1] * Er[2,-3]   → [a',phys;f]

function action(O::DenseProjectiveHamiltonian{2,1}, obj::MPSTensor{3})
    c = O.cache === nothing ? _init_proj_action_cache!(O, obj) : O.cache
    to = get_timer("action")
    objA = obj.A * one(c.TT)
    objW = MPSTensor(objA)
    TensorKit.zerovector!(c.acc)
    @timeit to "_proj_action1_mps" _proj_action1_contract!(c.cache, c.acc, objW, O.EnvL, O.EnvR)
    x = c.acc::MPSTensor{3}
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    return x
end

# 1-site 纯投影 MPS：El[a';a] · obj[a,phys;r] · Er[r;f] → [a',phys;f]
function _proj_action1_contract!(cache::Vector{Any}, acc::MPSTensor{3}, objW::MPSTensor{3}, ElW::DenseLeftEnvironmentTensor{2}, ErW::DenseRightEnvironmentTensor{2})
    objA = objW.A; El = ElW.A.A; Er = ErW.A.A
    if isempty(cache)
        c1 = objA * Er                                   # [a, phys; f]
        c2 = permute(c1, ((1,), (2,3)); copy=true)       # [a; phys, f]
        c3 = El * c2                                     # [a'; phys, f]
        empty!(cache); append!(cache, [c1, c2, c3])
    else
        TensorKit.mul!(cache[1], objA, Er, 1, 0)
        permute!(cache[2], cache[1], ((1,), (2,3)))
        TensorKit.mul!(cache[3], El, cache[2], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[3], ((1,2), (3,)), 1, 1)   # acc += [a', phys; f]
    return acc
end

# ====================== {3,1}/{3,2}：三体环境（中间算符 H）======================

function action(O::DenseProjectiveHamiltonian{3, 1}, obj::MPSTensor{3})
    c = O.cache === nothing ? _init_dense_action_cache!(O, obj) : O.cache
    to = get_timer("action")
    objA = obj.A * one(c.TT)
    objW = MPSTensor(objA)
    TensorKit.zerovector!(c.acc)
    @timeit to "_dense_action1" _dense_action1_contract!(c.cache, c.acc, objW, O.EnvL, O.H[1], O.EnvR)
    x = c.acc::MPSTensor{3}
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    return x
end

function action(O::DenseProjectiveHamiltonian{3, 2}, obj::CompositeMPSTensor{2, 4})
    c = O.cache === nothing ? _init_dense_action_cache!(O, obj) : O.cache
    to = get_timer("action")
    objA = obj.A * one(c.TT)
    objW = CompositeMPSTensor{2,4}(objA)
    TensorKit.zerovector!(c.acc)
    @timeit to "_dense_action2" _dense_action2_contract!(c.cache, c.acc, objW, O.EnvL, O.H[1], O.H[2], O.EnvR)
    x = c.acc::CompositeMPSTensor{2,4}
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    return x
end

# ====================== 1-site / 2-site 零分配链 =======================
# 索引约定以旧 @tensor 表达式为唯一规范（位置=裸张量维度序），逐条机械翻译为 permute!/mul!。
#   {3,1} MPS: tmp[-1,-2;-3] ≔ EnvL[-1,2,1] * h[-2,2,5,3] * obj[1,3,4] * EnvR[4,5,-3]
#   {3,2} MPS: tmp[-1,-2,-3;-4] ≔ EnvL[-1,2,1] * h1[-2,2,4,3] * h2[-3,4,7,5] * obj[1,3,5,6] * EnvR[6,7,-4]

# 1-site 稠密 MPS：El[a';β,a] · h[d,β;γ,phys] · obj[a,phys;r] · Er[r,γ;f] → [a',d;f]
function _dense_action1_contract!(cache::Vector{Any}, acc::MPSTensor{3}, objW::MPSTensor{3}, ElW::DenseLeftEnvironmentTensor{3}, hW::DenseMPOTensor{4}, ErW::DenseRightEnvironmentTensor{3})
    objA = objW.A; El = ElW.A.A; h = hW.A; Er = ErW.A.A
    if isempty(cache)
        c1 = permute(Er, ((1,), (2,3)); copy=true)       # [r; γ, f]
        c2 = objA * c1                                    # [a, phys; γ, f]
        c3 = permute(c2, ((3,2), (1,4)); copy=true)       # [γ, phys; a, f]
        c4 = h * c3                                       # [d, β; a, f]
        c5 = permute(c4, ((2,3), (1,4)); copy=true)       # [β, a; d, f]
        c6 = El * c5                                      # [a'; d, f]
        empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6])
    else
        permute!(cache[1], Er, ((1,), (2,3)))
        TensorKit.mul!(cache[2], objA, cache[1], 1, 0)
        permute!(cache[3], cache[2], ((3,2), (1,4)))
        TensorKit.mul!(cache[4], h, cache[3], 1, 0)
        permute!(cache[5], cache[4], ((2,3), (1,4)))
        TensorKit.mul!(cache[6], El, cache[5], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[6], ((1,2), (3,)), 1, 1)   # acc += [a', d; f]
    return acc
end

# 2-site 稠密 MPS：El[a';β₁,a] · h1[d₁,β₁;γ₁,phys₁] · h2[d₂,γ₁;γ₂,phys₂] · obj[a,phys₁,phys₂;r] · Er[r,γ₂;f] → [a',d₁,d₂;f]
function _dense_action2_contract!(cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::DenseLeftEnvironmentTensor{3}, h1W::DenseMPOTensor{4}, h2W::DenseMPOTensor{4}, ErW::DenseRightEnvironmentTensor{3})
    objA = objW.A; El = ElW.A.A; h1 = h1W.A; h2 = h2W.A; Er = ErW.A.A
    if isempty(cache)
        c1 = permute(Er, ((1,), (2,3)); copy=true)       # [r; γ₂, f]
        c2 = objA * c1                                    # [a, phys₁, phys₂; γ₂, f]
        c3 = permute(c2, ((4,3), (1,2,5)); copy=true)     # [γ₂, phys₂; a, phys₁, f]
        c4 = h2 * c3                                      # [d₂, γ₁; a, phys₁, f]
        c5 = permute(c4, ((2,4), (1,3,5)); copy=true)     # [γ₁, phys₁; d₂, a, f]
        c6 = h1 * c5                                      # [d₁, β₁; d₂, a, f]
        c7 = permute(c6, ((2,4), (1,3,5)); copy=true)     # [β₁, a; d₁, d₂, f]
        c8 = El * c7                                      # [a'; d₁, d₂, f]
        empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6, c7, c8])
    else
        permute!(cache[1], Er, ((1,), (2,3)))
        TensorKit.mul!(cache[2], objA, cache[1], 1, 0)
        permute!(cache[3], cache[2], ((4,3), (1,2,5)))
        TensorKit.mul!(cache[4], h2, cache[3], 1, 0)
        permute!(cache[5], cache[4], ((2,4), (1,3,5)))
        TensorKit.mul!(cache[6], h1, cache[5], 1, 0)
        permute!(cache[7], cache[6], ((2,4), (1,3,5)))
        TensorKit.mul!(cache[8], El, cache[7], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[8], ((1,2,3), (4,)), 1, 1)   # acc += [a', d₁, d₂; f]
    return acc
end
