# ====================== 稠密 cache action（MPO）======================
# DenseProjectiveHamiltonian 的零分配作用，MPO 侧：DenseMPOTensor{4} / CompositeMPOTensor{2,6}。

# ====================== {2,1}/{2,2}：两体环境纯投影（无算符，零分配）======================
# 旧 @tensor 参考：
#   {2,1} MPO: x[-1,-2;-3,-4] ≔ El[-2,1] * obj[-1,1,2,-4] * Er[2,-3]      → [phys,a';f,t]
#   {2,2} MPO: x[-1,-2,-3;-4,-5,-6] ≔ El[-3,1] * obj[-1,-2,1,2,-5,-6] * Er[2,-4] → [bR,bL,a';f,tR,tL]

function action(O::DenseProjectiveHamiltonian{2,1}, obj::DenseMPOTensor{4})
    c = O.cache === nothing ? _init_proj_action_cache!(O, obj) : O.cache
    to = get_timer("action")
    objA = obj.A * one(c.TT)
    objW = DenseMPOTensor{4}(objA)
    TensorKit.zerovector!(c.acc)
    @timeit to "_proj_action1_mpo" _proj_action1_contract!(c.cache, c.acc, objW, O.EnvL, O.EnvR)
    x = c.acc::DenseMPOTensor{4}
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    return x
end

function action(O::DenseProjectiveHamiltonian{2,2}, obj::CompositeMPOTensor{2,6})
    c = O.cache === nothing ? _init_proj_action_cache!(O, obj) : O.cache
    to = get_timer("action")
    objA = obj.A * one(c.TT)
    objW = CompositeMPOTensor{2,6}(objA)
    TensorKit.zerovector!(c.acc)
    @timeit to "_proj_action2_mpo" _proj_action2_contract!(c.cache, c.acc, objW, O.EnvL, O.EnvR)
    x = c.acc::CompositeMPOTensor{2,6}
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    return x
end

# 1-site 纯投影 MPO：El[a';a] · obj[phys,a;r,t] · Er[r;f] → [phys,a';f,t]
function _proj_action1_contract!(cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::DenseLeftEnvironmentTensor{2}, ErW::DenseRightEnvironmentTensor{2})
    objA = objW.A; El = ElW.A.A; Er = ErW.A.A
    if isempty(cache)
        c1 = permute(objA, ((1,2,4), (3,)); copy=true)   # [phys, a, t; r]
        c2 = c1 * Er                                      # [phys, a, t; f]
        c3 = permute(c2, ((2,), (1,3,4)); copy=true)      # [a; phys, t, f]
        c4 = El * c3                                      # [a'; phys, t, f]
        empty!(cache); append!(cache, [c1, c2, c3, c4])
    else
        permute!(cache[1], objA, ((1,2,4), (3,)))
        TensorKit.mul!(cache[2], cache[1], Er, 1, 0)
        permute!(cache[3], cache[2], ((2,), (1,3,4)))
        TensorKit.mul!(cache[4], El, cache[3], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[4], ((2,1), (4,3)), 1, 1)   # acc += [phys, a'; f, t]
    return acc
end

# 2-site 纯投影 MPO：El[a';a] · obj[bR,bL,a;r,tR,tL] · Er[r;f] → [bR,bL,a';f,tR,tL]
function _proj_action2_contract!(cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::DenseLeftEnvironmentTensor{2}, ErW::DenseRightEnvironmentTensor{2})
    objA = objW.A; El = ElW.A.A; Er = ErW.A.A
    if isempty(cache)
        c1 = permute(objA, ((1,2,3,5,6), (4,)); copy=true)   # [bR, bL, a, tR, tL; r]
        c2 = c1 * Er                                          # [bR, bL, a, tR, tL; f]
        c3 = permute(c2, ((3,), (1,2,4,5,6)); copy=true)      # [a; bR, bL, tR, tL, f]
        c4 = El * c3                                          # [a'; bR, bL, tR, tL, f]
        empty!(cache); append!(cache, [c1, c2, c3, c4])
    else
        permute!(cache[1], objA, ((1,2,3,5,6), (4,)))
        TensorKit.mul!(cache[2], cache[1], Er, 1, 0)
        permute!(cache[3], cache[2], ((3,), (1,2,4,5,6)))
        TensorKit.mul!(cache[4], El, cache[3], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[4], ((2,3,1), (6,4,5)), 1, 1)   # acc += [bR, bL, a'; f, tR, tL]
    return acc
end

# ====================== {3,1}/{3,2}：三体环境（中间算符 H）======================

function action(O::DenseProjectiveHamiltonian{3, 2}, obj::CompositeMPOTensor{2, 6})
    c = O.cache === nothing ? _init_dense_action_cache!(O, obj) : O.cache
    to = get_timer("action")
    objA = obj.A * one(c.TT)
    objW = CompositeMPOTensor{2,6}(objA)
    TensorKit.zerovector!(c.acc)
    @timeit to "_dense_action2_mpo" _dense_action2_contract!(c.cache, c.acc, objW, O.EnvL, O.H[1], O.H[2], O.EnvR)
    x = c.acc::CompositeMPOTensor{2,6}
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    return x
end

function action(O::DenseProjectiveHamiltonian{3, 1}, obj::DenseMPOTensor{4})
    c = O.cache === nothing ? _init_dense_action_cache!(O, obj) : O.cache
    to = get_timer("action")
    objA = obj.A * one(c.TT)
    objW = DenseMPOTensor{4}(objA)
    TensorKit.zerovector!(c.acc)
    @timeit to "_dense_action1_mpo" _dense_action1_contract!(c.cache, c.acc, objW, O.EnvL, O.H[1], O.EnvR)
    x = c.acc::DenseMPOTensor{4}
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    return x
end

# ====================== 1-site / 2-site 零分配链 =======================

# 1-site 稠密 MPO：El[a';β,a] · h[d,β;γ,phys] · obj[phys,a;r,t] · Er[r,γ;f] → [d,a';f,t]
function _dense_action1_contract!(cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::DenseLeftEnvironmentTensor{3}, hW::DenseMPOTensor{4}, ErW::DenseRightEnvironmentTensor{3})
    objA = objW.A; El = ElW.A.A; h = hW.A; Er = ErW.A.A
    if isempty(cache)
        c1 = permute(objA, ((1,2,4), (3,)); copy=true)   # [phys, a, t; r]
        c2 = permute(Er, ((1,), (2,3)); copy=true)       # [r; γ, f]
        c3 = c1 * c2                                      # [phys, a, t; γ, f]
        c4 = permute(c3, ((4,1), (2,3,5)); copy=true)     # [γ, phys; a, t, f]
        c5 = h * c4                                       # [d, β; a, t, f]
        c6 = permute(c5, ((2,3), (1,4,5)); copy=true)     # [β, a; d, t, f]
        c7 = El * c6                                      # [a'; d, t, f]
        empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6, c7])
    else
        permute!(cache[1], objA, ((1,2,4), (3,)))
        permute!(cache[2], Er, ((1,), (2,3)))
        TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
        permute!(cache[4], cache[3], ((4,1), (2,3,5)))
        TensorKit.mul!(cache[5], h, cache[4], 1, 0)
        permute!(cache[6], cache[5], ((2,3), (1,4,5)))
        TensorKit.mul!(cache[7], El, cache[6], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[7], ((2,1), (4,3)), 1, 1)   # acc += [d, a'; f, t]
    return acc
end

# 1-site 稠密 MPO（AdjointMPO）：El[a',β;a] · h[γ,phys;d,β] · obj[phys,a;r,t] · Er[r;γ,β'] → [phys,a';β',t]
# 环境左右腿对调（El=(1,2)←3，Er=1←(2,3)），物理腿缩并是 h.d(dom)↔obj.phys(codom)，输出 h.phys(codom)。
function _dense_action1_contract!(cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::DenseLeftEnvironmentTensor{3}, hW::AdjointMPOTensor{4}, ErW::DenseRightEnvironmentTensor{3})
    objA = objW.A; El = ElW.A.A; h = hW.A; Er = ErW.A.A
    if isempty(cache)
        Erp  = permute(Er, ((1,), (2,3)); copy=true)      # [r; γ, β']（Er 去对偶，γ 落 domain）
        El1  = permute(El, ((2,), (1,3)); copy=true)       # [β; a'', a]
        h1   = permute(h, ((1,2,3), (4,)); copy=true)      # [γ, phys, d'; β]
        c1   = h1 * El1                                    # [γ, phys, d'; a'', a]
        obj1 = permute(objA, ((1,2,4), (3,)); copy=true)   # [phys, a, t'; r]
        c2   = obj1 * Erp                                  # [phys, a, t'; γ, β']
        A    = permute(c1, ((2,4), (1,3,5)); copy=true)    # [phys, a'''; γ', d'', a]
        B    = permute(c2, ((4,1,2), (3,5)); copy=true)    # [γ', phys, a; t, β']
        c3   = A * B                                       # [phys, a'''; t, β']
        empty!(cache); append!(cache, [Erp, El1, h1, c1, obj1, c2, A, B, c3])
    else
        permute!(cache[1], Er, ((1,), (2,3)))
        permute!(cache[2], El, ((2,), (1,3)))
        permute!(cache[3], h, ((1,2,3), (4,)))
        TensorKit.mul!(cache[4], cache[3], cache[2], 1, 0)
        permute!(cache[5], objA, ((1,2,4), (3,)))
        TensorKit.mul!(cache[6], cache[5], cache[1], 1, 0)
        permute!(cache[7], cache[4], ((2,4), (1,3,5)))
        permute!(cache[8], cache[6], ((4,1,2), (3,5)))
        TensorKit.mul!(cache[9], cache[7], cache[8], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[9], ((1,2), (4,3)), 1, 1)   # acc += [phys, a'; β', t]
    return acc
end

# 2-site 稠密 MPO：El[a';β₁,a] · h1[d₁,β₁;γ₁,phys₁] · h2[d₂,γ₁;γ₂,phys₂] · obj[bR,bL,a;r,tR,tL] · Er[r,γ₂;f] → [d₂,d₁,a';f,tR,tL]
function _dense_action2_contract!(cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::DenseLeftEnvironmentTensor{3}, h1W::DenseMPOTensor{4}, h2W::DenseMPOTensor{4}, ErW::DenseRightEnvironmentTensor{3})
    objA = objW.A; El = ElW.A.A; h1 = h1W.A; h2 = h2W.A; Er = ErW.A.A
    if isempty(cache)
        c1 = permute(objA, ((1,2,3,5,6), (4,)); copy=true)   # [bR, bL, a, tR, tL; r]
        c2 = permute(Er, ((1,), (2,3)); copy=true)           # [r; γ₂, f]
        c3 = c1 * c2                                          # [bR, bL, a, tR, tL; γ₂, f]
        c4 = permute(c3, ((6,1), (2,3,4,5,7)); copy=true)     # [γ₂, bR; bL, a, tR, tL, f]
        c5 = h2 * c4                                          # [d₂, γ₁; bL, a, tR, tL, f]
        c6 = permute(c5, ((2,3), (1,4,5,6,7)); copy=true)     # [γ₁, bL; d₂, a, tR, tL, f]
        c7 = h1 * c6                                          # [d₁, β₁; d₂, a, tR, tL, f]
        c8 = permute(c7, ((2,4), (1,3,5,6,7)); copy=true)     # [β₁, a; d₁, d₂, tR, tL, f]
        c9 = El * c8                                          # [a'; d₁, d₂, tR, tL, f]
        empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6, c7, c8, c9])
    else
        permute!(cache[1], objA, ((1,2,3,5,6), (4,)))
        permute!(cache[2], Er, ((1,), (2,3)))
        TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
        permute!(cache[4], cache[3], ((6,1), (2,3,4,5,7)))
        TensorKit.mul!(cache[5], h2, cache[4], 1, 0)
        permute!(cache[6], cache[5], ((2,3), (1,4,5,6,7)))
        TensorKit.mul!(cache[7], h1, cache[6], 1, 0)
        permute!(cache[8], cache[7], ((2,4), (1,3,5,6,7)))
        TensorKit.mul!(cache[9], El, cache[8], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[9], ((3,2,1), (6,4,5)), 1, 1)   # acc += [d₂, d₁, a'; f, tR, tL]
    return acc
end

# 2-site 稠密 MPO（AdjointMPO）：El[a',β₁;a] · h1[γ₁,phys₁;d₁,β₁] · h2[γ₂,phys₂;d₂,γ₁] · obj[bR,bL,a;r,tR,tL] · Er[r;γ₂,β'] → [phys₂,phys₁,a';β',tR,tL]
# 先把 h1·h2 沿内键 γ₁ 合成 2-site 算符 H=[γ₂,phys₂,phys₁;d₂,d₁,β₁]，再套 1-site 的「El·H + obj·Er → 合并」结构。
function _dense_action2_contract!(cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::DenseLeftEnvironmentTensor{3}, h1W::AdjointMPOTensor{4}, h2W::AdjointMPOTensor{4}, ErW::DenseRightEnvironmentTensor{3})
    objA = objW.A; El = ElW.A.A; h1 = h1W.A; h2 = h2W.A; Er = ErW.A.A
    if isempty(cache)
        h1a  = permute(h1, ((1,), (2,3,4)); copy=true)       # [γ₁; phys₁', d₁, β₁]
        h2a  = permute(h2, ((1,2,3), (4,)); copy=true)       # [γ₂, phys₂, d₂'; γ₁]
        Htmp = h2a * h1a                                      # [γ₂, phys₂, d₂'; phys₁', d₁, β₁]
        H    = permute(Htmp, ((1,2,4), (3,5,6)); copy=true)  # [γ₂, phys₂, phys₁; d₂, d₁, β₁]
        Erp  = permute(Er, ((1,), (2,3)); copy=true)         # [r; γ₂, β']
        El1  = permute(El, ((2,), (1,3)); copy=true)          # [β₁; a'', a]
        H1   = permute(H, ((1,2,3,4,5), (6,)); copy=true)    # [γ₂, phys₂, phys₁, d₂', d₁'; β₁]
        c1   = H1 * El1                                       # [γ₂, phys₂, phys₁, d₂', d₁'; a'', a]
        obj1 = permute(objA, ((1,2,3,5,6), (4,)); copy=true)  # [bR, bL, a, tR', tL'; r]
        c2   = obj1 * Erp                                     # [bR, bL, a, tR', tL'; γ₂, β']
        A    = permute(c1, ((2,3,6), (1,4,5,7)); copy=true)  # [phys₂, phys₁, a'''; γ₂', d₂'', d₁'', a]
        B    = permute(c2, ((6,1,2,3), (4,5,7)); copy=true)  # [γ₂', bR, bL, a; tR, tL, β']
        c3   = A * B                                          # [phys₂, phys₁, a'''; tR, tL, β']
        empty!(cache); append!(cache, [h1a, h2a, Htmp, H, Erp, El1, H1, c1, obj1, c2, A, B, c3])
    else
        permute!(cache[1], h1, ((1,), (2,3,4)))
        permute!(cache[2], h2, ((1,2,3), (4,)))
        TensorKit.mul!(cache[3], cache[2], cache[1], 1, 0)
        permute!(cache[4], cache[3], ((1,2,4), (3,5,6)))
        permute!(cache[5], Er, ((1,), (2,3)))
        permute!(cache[6], El, ((2,), (1,3)))
        permute!(cache[7], cache[4], ((1,2,3,4,5), (6,)))
        TensorKit.mul!(cache[8], cache[7], cache[6], 1, 0)
        permute!(cache[9], objA, ((1,2,3,5,6), (4,)))
        TensorKit.mul!(cache[10], cache[9], cache[5], 1, 0)
        permute!(cache[11], cache[8], ((2,3,6), (1,4,5,7)))
        permute!(cache[12], cache[10], ((6,1,2,3), (4,5,7)))
        TensorKit.mul!(cache[13], cache[11], cache[12], 1, 0)
    end
    TensorKit.add_permute!(acc.A, cache[13], ((1,2,3), (6,4,5)), 1, 1)   # acc += [phys₂, phys₁, a'; β', tR, tL]
    return acc
end
