# ====================== 2-site MPS 零分配 action（缓存 + 原地 mul!/permute! + β=1 融合累加）======================
# 与 action1.jl 同构：每 validind 一套中间缓冲（惰性分配，跨 action 调用复用）；每 worker 一个累加器。
# 统一函数名 _action2_contract，按包装类型多重分发（动态派发）：El/Er 是 LeftEnvironmentTensor{2/3} / RightEnvironmentTensor{2/3}，
# hl/hr 是 LocalOperator{1,1}/{1,2}/{2,1} 或 IdentityOperator{1}。方法内部取 .A 得到裸张量后走原地缩并链。
# 索引约定照搬 src/Hamiltonian/contract.jl 的 _action2_contract：
#   obj[a,p1,p2,r]（codom (a,p1,p2), dom (r)），结果 [a',d1',d2';f]（Id 情形 d'→p）
# 算符布局：{1,1}=[d;p]，{1,2}=[d;g,p]（aux g 在 domain），{2,1}=[d,g;p]（aux g 在 codomain）。
# 环境布局：El{2}=[a';a]，El{3}=[a';b,a]；Er{2}=[r;f]，Er{3}=[r,b;f]。
# 末步统一 TensorKit.mul!(acc, cache[end], Er, wm, 1)：cache[end]=[a',d1',d2';r]（Er{3} 情形为 [a',d1',d2';r,b]）。
# 缓存分配用 permute(...; copy=true)（分配版 permute 可能返回共享内存 view，会被后续原地写破坏）。
# @timeit 用字面量标签（编译期常量），to 为每 worker 线程本地计时器，无锁争用。

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,1}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_11_11_2" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3,4)); copy=true)   # [a; p1,p2,r]
            c2 = El * c1                                     # [a'; p1,p2,r]
            c3 = permute(c2, ((2,), (1,3,4)); copy=true)     # [p1; a',p2,r]
            c4 = hl * c3                                     # [d1; a',p2,r]
            c5 = permute(c4, ((3,), (1,2,4)); copy=true)     # [p2; d1,a',r]
            c6 = hr * c5                                     # [d2; d1,a',r]
            c7 = permute(c6, ((3,2,1), (4,)); copy=true)     # [a',d1,d2; r]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7])
        else
            permute!(cache[1], objA, ((1,), (2,3,4)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3,4)))
            TensorKit.mul!(cache[4], hl, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((3,), (1,2,4)))
            TensorKit.mul!(cache[6], hr, cache[5], 1, 0)
            permute!(cache[7], cache[6], ((3,2,1), (4,)))
        end
        TensorKit.mul!(acc.A, cache[7], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_00_11_2" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3,4)); copy=true)
            c2 = El * c1
            c3 = permute(c2, ((3,), (1,2,4)); copy=true)
            c4 = hr * c3
            c5 = permute(c4, ((2,3,1), (4,)); copy=true)
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5])
        else
            permute!(cache[1], objA, ((1,), (2,3,4)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((3,), (1,2,4)))
            TensorKit.mul!(cache[4], hr, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((2,3,1), (4,)))
        end
        TensorKit.mul!(acc.A, cache[5], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,1}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; Er = ErW.A
    @timeit to "_action2_2_2_11_00_2" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3,4)); copy=true)
            c2 = El * c1
            c3 = permute(c2, ((2,), (1,3,4)); copy=true)
            c4 = hl * c3
            c5 = permute(c4, ((2,1,3), (4,)); copy=true)
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5])
        else
            permute!(cache[1], objA, ((1,), (2,3,4)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3,4)))
            TensorKit.mul!(cache[4], hl, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((2,1,3), (4,)))
        end
        TensorKit.mul!(acc.A, cache[5], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; Er = ErW.A
    @timeit to "_action2_2_2_00_00_2" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3,4)); copy=true)
            c2 = El * c1
            c3 = permute(c2, ((1,2,3), (4,)); copy=true)
            empty!(cache); append!(cache, [c1,c2,c3])
        else
            permute!(cache[1], objA, ((1,), (2,3,4)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((1,2,3), (4,)))
        end
        TensorKit.mul!(acc.A, cache[3], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,2}, hrW::LocalOperator{2,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_12_21_2" begin
        if isempty(cache)
            c1 = permute(hl, ((1,2), (3,)); copy=true)       # [d1,g; p1]
            c2 = permute(objA, ((2,), (1,3,4)); copy=true)   # [p1; a,p2,r]
            c3 = c1 * c2                                     # [d1,g; a,p2,r]
            c4 = permute(c3, ((1,3,5), (2,4)); copy=true)    # [d1,a,r; g,p2]
            c5 = permute(hr, ((2,3), (1,)); copy=true)       # [g,p2; d2]
            c6 = c4 * c5                                     # [d1,a,r; d2]
            c7 = permute(c6, ((2,), (1,3,4)); copy=true)     # [a; d1,r,d2]
            c8 = El * c7                                     # [a'; d1,r,d2]
            c9 = permute(c8, ((1,2,4), (3,)); copy=true)     # [a',d1,d2; r]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9])
        else
            permute!(cache[1], hl, ((1,2), (3,)))
            permute!(cache[2], objA, ((2,), (1,3,4)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3,5), (2,4)))
            permute!(cache[5], hr, ((2,3), (1,)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((2,), (1,3,4)))
            TensorKit.mul!(cache[8], El, cache[7], 1, 0)
            permute!(cache[9], cache[8], ((1,2,4), (3,)))
        end
        TensorKit.mul!(acc.A, cache[9], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,2}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_12_11_3" begin
        if isempty(cache)
            c1 = permute(hl, ((1,2), (3,)); copy=true)       # [d1,g; p1]
            c2 = permute(objA, ((2,), (1,3,4)); copy=true)   # [p1; a,p2,r]
            c3 = c1 * c2                                     # [d1,g; a,p2,r]
            c4 = permute(c3, ((4,), (1,2,3,5)); copy=true)   # [p2; d1,g,a,r]
            c5 = hr * c4                                     # [d2; d1,g,a,r]
            c6 = permute(c5, ((4,), (1,2,3,5)); copy=true)   # [a; d2,d1,g,r]
            c7 = El * c6                                     # [a'; d2,d1,g,r]
            c8 = permute(c7, ((1,3,2), (5,4)); copy=true)    # [a',d1,d2; r,g]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8])
        else
            permute!(cache[1], hl, ((1,2), (3,)))
            permute!(cache[2], objA, ((2,), (1,3,4)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((4,), (1,2,3,5)))
            TensorKit.mul!(cache[5], hr, cache[4], 1, 0)
            permute!(cache[6], cache[5], ((4,), (1,2,3,5)))
            TensorKit.mul!(cache[7], El, cache[6], 1, 0)
            permute!(cache[8], cache[7], ((1,3,2), (5,4)))
        end
        TensorKit.mul!(acc.A, cache[8], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,2}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; Er = ErW.A
    @timeit to "_action2_2_2_12_00_3" begin
        if isempty(cache)
            c1 = permute(hl, ((1,2), (3,)); copy=true)       # [d1,g; p1]
            c2 = permute(objA, ((2,), (1,3,4)); copy=true)   # [p1; a,p2,r]
            c3 = c1 * c2                                     # [d1,g; a,p2,r]
            c4 = permute(c3, ((3,), (1,2,4,5)); copy=true)   # [a; d1,g,p2,r]
            c5 = El * c4                                     # [a'; d1,g,p2,r]
            c6 = permute(c5, ((1,2,4), (5,3)); copy=true)    # [a',d1,p2; r,g]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6])
        else
            permute!(cache[1], hl, ((1,2), (3,)))
            permute!(cache[2], objA, ((2,), (1,3,4)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((3,), (1,2,4,5)))
            TensorKit.mul!(cache[5], El, cache[4], 1, 0)
            permute!(cache[6], cache[5], ((1,2,4), (5,3)))
        end
        TensorKit.mul!(acc.A, cache[6], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,1}, hrW::LocalOperator{1,2}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_11_12_3" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3,4)); copy=true)   # [p1; a,p2,r]
            c2 = hl * c1                                     # [d1; a,p2,r]
            c3 = permute(c2, ((3,), (1,2,4)); copy=true)     # [p2; d1,a,r]
            c4 = permute(hr, ((1,2), (3,)); copy=true)       # [d2,g; p2]
            c5 = c4 * c3                                     # [d2,g; d1,a,r]
            c6 = permute(c5, ((4,), (1,2,3,5)); copy=true)   # [a; d2,g,d1,r]
            c7 = El * c6                                     # [a'; d2,g,d1,r]
            c8 = permute(c7, ((1,4,2), (5,3)); copy=true)    # [a',d1,d2; r,g]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8])
        else
            permute!(cache[1], objA, ((2,), (1,3,4)))
            TensorKit.mul!(cache[2], hl, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((3,), (1,2,4)))
            permute!(cache[4], hr, ((1,2), (3,)))
            TensorKit.mul!(cache[5], cache[4], cache[3], 1, 0)
            permute!(cache[6], cache[5], ((4,), (1,2,3,5)))
            TensorKit.mul!(cache[7], El, cache[6], 1, 0)
            permute!(cache[8], cache[7], ((1,4,2), (5,3)))
        end
        TensorKit.mul!(acc.A, cache[8], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, hrW::LocalOperator{1,2}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_00_12_3" begin
        if isempty(cache)
            c1 = permute(objA, ((3,), (1,2,4)); copy=true)   # [p2; a,p1,r]
            c2 = permute(hr, ((1,2), (3,)); copy=true)       # [d2,g; p2]
            c3 = c2 * c1                                     # [d2,g; a,p1,r]
            c4 = permute(c3, ((3,), (1,2,4,5)); copy=true)   # [a; d2,g,p1,r]
            c5 = El * c4                                     # [a'; d2,g,p1,r]
            c6 = permute(c5, ((1,4,2), (5,3)); copy=true)    # [a',p1,d2; r,g]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6])
        else
            permute!(cache[1], objA, ((3,), (1,2,4)))
            permute!(cache[2], hr, ((1,2), (3,)))
            TensorKit.mul!(cache[3], cache[2], cache[1], 1, 0)
            permute!(cache[4], cache[3], ((3,), (1,2,4,5)))
            TensorKit.mul!(cache[5], El, cache[4], 1, 0)
            permute!(cache[6], cache[5], ((1,4,2), (5,3)))
        end
        TensorKit.mul!(acc.A, cache[6], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{3}, hlW::LocalOperator{2,1}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_3_21_11_2" begin
        if isempty(cache)
            c1 = permute(El, ((1,3), (2,)); copy=true)       # [a',a; b]
            c2 = permute(hl, ((2,), (1,3)); copy=true)       # [b; d1,p1]
            c3 = c1 * c2                                     # [a',a; d1,p1]
            c4 = permute(c3, ((1,3), (2,4)); copy=true)      # [a',d1; a,p1]
            c5 = permute(objA, ((1,2), (3,4)); copy=true)    # [a,p1; p2,r]
            c6 = c4 * c5                                     # [a',d1; p2,r]
            c7 = permute(c6, ((3,), (1,2,4)); copy=true)     # [p2; a',d1,r]
            c8 = hr * c7                                     # [d2; a',d1,r]
            c9 = permute(c8, ((2,3,1), (4,)); copy=true)     # [a',d1,d2; r]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9])
        else
            permute!(cache[1], El, ((1,3), (2,)))
            permute!(cache[2], hl, ((2,), (1,3)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (2,4)))
            permute!(cache[5], objA, ((1,2), (3,4)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((3,), (1,2,4)))
            TensorKit.mul!(cache[8], hr, cache[7], 1, 0)
            permute!(cache[9], cache[8], ((2,3,1), (4,)))
        end
        TensorKit.mul!(acc.A, cache[9], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{3}, hlW::LocalOperator{2,1}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; Er = ErW.A
    @timeit to "_action2_2_3_21_00_2" begin
        if isempty(cache)
            c1 = permute(El, ((1,3), (2,)); copy=true)       # [a',a; b]
            c2 = permute(hl, ((2,), (1,3)); copy=true)       # [b; d1,p1]
            c3 = c1 * c2                                     # [a',a; d1,p1]
            c4 = permute(c3, ((1,3), (2,4)); copy=true)      # [a',d1; a,p1]
            c5 = permute(objA, ((1,2), (3,4)); copy=true)    # [a,p1; p2,r]
            c6 = c4 * c5                                     # [a',d1; p2,r]
            c7 = permute(c6, ((1,2,3), (4,)); copy=true)     # [a',d1,p2; r]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7])
        else
            permute!(cache[1], El, ((1,3), (2,)))
            permute!(cache[2], hl, ((2,), (1,3)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (2,4)))
            permute!(cache[5], objA, ((1,2), (3,4)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((1,2,3), (4,)))
        end
        TensorKit.mul!(acc.A, cache[7], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{3}, hlW::LocalOperator{1,1}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_3_11_11_3" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3,4)); copy=true)   # [p1; a,p2,r]
            c2 = hl * c1                                     # [d1; a,p2,r]
            c3 = permute(c2, ((3,), (1,2,4)); copy=true)     # [p2; d1,a,r]
            c4 = hr * c3                                     # [d2; d1,a,r]
            c5 = permute(c4, ((3,), (1,2,4)); copy=true)     # [a; d2,d1,r]
            c6 = permute(El, ((1,2), (3,)); copy=true)       # [a',b; a]（El=[a';b,a] → b 进 codom）
            c7 = c6 * c5                                     # [a',b; d2,d1,r]
            c8 = permute(c7, ((1,4,3), (5,2)); copy=true)    # [a',d1,d2; r,b]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8])
        else
            permute!(cache[1], objA, ((2,), (1,3,4)))
            TensorKit.mul!(cache[2], hl, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((3,), (1,2,4)))
            TensorKit.mul!(cache[4], hr, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((3,), (1,2,4)))
            permute!(cache[6], El, ((1,2), (3,)))
            TensorKit.mul!(cache[7], cache[6], cache[5], 1, 0)
            permute!(cache[8], cache[7], ((1,4,3), (5,2)))
        end
        TensorKit.mul!(acc.A, cache[8], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; Er = ErW.A
    @timeit to "_action2_2_3_00_00_3" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3,4)); copy=true)   # [a; p1,p2,r]
            c2 = permute(El, ((1,2), (3,)); copy=true)       # [a',b; a]
            c3 = c2 * c1                                     # [a',b; p1,p2,r]
            c4 = permute(c3, ((1,3,4), (5,2)); copy=true)    # [a',p1,p2; r,b]
            empty!(cache); append!(cache, [c1,c2,c3,c4])
        else
            permute!(cache[1], objA, ((1,), (2,3,4)))
            permute!(cache[2], El, ((1,2), (3,)))
            TensorKit.mul!(cache[3], cache[2], cache[1], 1, 0)
            permute!(cache[4], cache[3], ((1,3,4), (5,2)))
        end
        TensorKit.mul!(acc.A, cache[4], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{3}, hlW::LocalOperator{1,1}, hrW::LocalOperator{2,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_3_11_21_2" begin
        if isempty(cache)
            c1 = permute(El, ((1,3), (2,)); copy=true)       # [a',a; b]
            c2 = permute(hr, ((2,), (1,3)); copy=true)       # [b; d2,p2]
            c3 = c1 * c2                                     # [a',a; d2,p2]
            c4 = permute(c3, ((1,3), (2,4)); copy=true)      # [a',d2; a,p2]
            c5 = permute(objA, ((1,3), (2,4)); copy=true)    # [a,p2; p1,r]
            c6 = c4 * c5                                     # [a',d2; p1,r]
            c7 = permute(c6, ((3,), (1,2,4)); copy=true)     # [p1; a',d2,r]
            c8 = hl * c7                                     # [d1; a',d2,r]
            c9 = permute(c8, ((2,1,3), (4,)); copy=true)     # [a',d1,d2; r]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9])
        else
            permute!(cache[1], El, ((1,3), (2,)))
            permute!(cache[2], hr, ((2,), (1,3)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (2,4)))
            permute!(cache[5], objA, ((1,3), (2,4)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((3,), (1,2,4)))
            TensorKit.mul!(cache[8], hl, cache[7], 1, 0)
            permute!(cache[9], cache[8], ((2,1,3), (4,)))
        end
        TensorKit.mul!(acc.A, cache[9], Er, wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPSTensor{2,4}, objW::CompositeMPSTensor{2,4}, ElW::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, hrW::LocalOperator{2,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_3_00_21_2" begin
        if isempty(cache)
            c1 = permute(El, ((1,3), (2,)); copy=true)       # [a',a; b]
            c2 = permute(hr, ((2,), (1,3)); copy=true)       # [b; d2,p2]
            c3 = c1 * c2                                     # [a',a; d2,p2]
            c4 = permute(c3, ((1,3), (2,4)); copy=true)      # [a',d2; a,p2]
            c5 = permute(objA, ((1,3), (2,4)); copy=true)    # [a,p2; p1,r]
            c6 = c4 * c5                                     # [a',d2; p1,r]
            c7 = permute(c6, ((1,3,2), (4,)); copy=true)     # [a',p1,d2; r]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7])
        else
            permute!(cache[1], El, ((1,3), (2,)))
            permute!(cache[2], hr, ((2,), (1,3)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (2,4)))
            permute!(cache[5], objA, ((1,3), (2,4)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((1,3,2), (4,)))
        end
        TensorKit.mul!(acc.A, cache[7], Er, wm, 1)
    end
    return acc
end

# ====================== 2-site MPO 零分配 action（CompositeMPOTensor{2,6}）======================
# 与 MPS 同构：每 validind 一套中间缓冲 + 每 worker 一个累加器。
# objA = CompositeMPOTensor{2,6}.A = [下右(bR), 下左(bL), 左(a); 右(r), 上右(tR), 上左(tL)]（codom (bR,bL,a)，dom (r,tR,tL)）。
# hr 作用于 bR（site2），hl 作用于 bL（site1）；结果 x = [d2, d1, a'; f, tR, tL]。
# 注意物理腿顺序是 site2 在前、site1 在后，且左键 a' 在 codom 最后（与 MPS 的 [a',d1,d2;f] 不同）。
# 带 aux 的链末步 mul! + add_permute!（把 phys/键顺序摆正并融合累加）。

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,1}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_11_11_2_m" begin
        if isempty(cache)
            c1 = permute(objA, ((3,), (1,2,4,5,6)); copy=true)   # [a; bR,bL,r,tR,tL]
            c2 = El * c1                                         # [a'; bR,bL,r,tR,tL]
            c3 = permute(c2, ((3,), (1,2,4,5,6)); copy=true)     # [bL; a',bR,r,tR,tL]
            c4 = hl * c3                                         # [d1; a',bR,r,tR,tL]
            c5 = permute(c4, ((3,), (1,2,4,5,6)); copy=true)     # [bR; d1,a',r,tR,tL]
            c6 = hr * c5                                         # [d2; d1,a',r,tR,tL]
            c7 = permute(c6, ((1,2,3,5,6), (4,)); copy=true)     # [d2,d1,a',tR,tL; r]
            c8 = c7 * Er                                         # [d2,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8])
        else
            permute!(cache[1], objA, ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[4], hl, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[6], hr, cache[5], 1, 0)
            permute!(cache[7], cache[6], ((1,2,3,5,6), (4,)))
            TensorKit.mul!(cache[8], cache[7], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[8], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_00_11_2_m" begin
        if isempty(cache)
            c1 = permute(objA, ((3,), (1,2,4,5,6)); copy=true)   # [a; bR,bL,r,tR,tL]
            c2 = El * c1                                         # [a'; bR,bL,r,tR,tL]
            c3 = permute(c2, ((2,), (1,3,4,5,6)); copy=true)     # [bR; a',bL,r,tR,tL]
            c4 = hr * c3                                         # [d2; a',bL,r,tR,tL]
            c5 = permute(c4, ((1,3,2,5,6), (4,)); copy=true)     # [d2,bL,a',tR,tL; r]
            c6 = c5 * Er                                         # [d2,bL,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6])
        else
            permute!(cache[1], objA, ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3,4,5,6)))
            TensorKit.mul!(cache[4], hr, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((1,3,2,5,6), (4,)))
            TensorKit.mul!(cache[6], cache[5], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[6], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,1}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; Er = ErW.A
    @timeit to "_action2_2_2_11_00_2_m" begin
        if isempty(cache)
            c1 = permute(objA, ((3,), (1,2,4,5,6)); copy=true)   # [a; bR,bL,r,tR,tL]
            c2 = El * c1                                         # [a'; bR,bL,r,tR,tL]
            c3 = permute(c2, ((3,), (1,2,4,5,6)); copy=true)     # [bL; a',bR,r,tR,tL]
            c4 = hl * c3                                         # [d1; a',bR,r,tR,tL]
            c5 = permute(c4, ((3,1,2,5,6), (4,)); copy=true)     # [bR,d1,a',tR,tL; r]
            c6 = c5 * Er                                         # [bR,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6])
        else
            permute!(cache[1], objA, ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[4], hl, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((3,1,2,5,6), (4,)))
            TensorKit.mul!(cache[6], cache[5], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[6], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; Er = ErW.A
    @timeit to "_action2_2_2_00_00_2_m" begin
        if isempty(cache)
            c1 = permute(objA, ((3,), (1,2,4,5,6)); copy=true)   # [a; bR,bL,r,tR,tL]
            c2 = El * c1                                         # [a'; bR,bL,r,tR,tL]
            c3 = permute(c2, ((2,3,1,5,6), (4,)); copy=true)     # [bR,bL,a',tR,tL; r]
            c4 = c3 * Er                                         # [bR,bL,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4])
        else
            permute!(cache[1], objA, ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,3,1,5,6), (4,)))
            TensorKit.mul!(cache[4], cache[3], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[4], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,2}, hrW::LocalOperator{2,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_12_21_2_m" begin
        if isempty(cache)
            c1 = permute(hl, ((1,2), (3,)); copy=true)          # [d1,g; bL]
            c2 = permute(objA, ((2,), (1,3,4,5,6)); copy=true)  # [bL; bR,a,r,tR,tL]
            c3 = c1 * c2                                        # [d1,g; bR,a,r,tR,tL]
            c4 = permute(c3, ((1,4,5,6,7), (2,3)); copy=true)   # [d1,a,r,tR,tL; g,bR]
            c5 = permute(hr, ((2,3), (1,)); copy=true)          # [g,bR; d2]
            c6 = c4 * c5                                        # [d1,a,r,tR,tL; d2]
            c7 = permute(c6, ((2,), (1,3,4,5,6)); copy=true)    # [a; d1,r,tR,tL,d2]
            c8 = El * c7                                        # [a'; d1,r,tR,tL,d2]
            c9 = permute(c8, ((6,2,1,4,5), (3,)); copy=true)    # [d2,d1,a',tR,tL; r]
            c10 = c9 * Er                                       # [d2,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])
        else
            permute!(cache[1], hl, ((1,2), (3,)))
            permute!(cache[2], objA, ((2,), (1,3,4,5,6)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,4,5,6,7), (2,3)))
            permute!(cache[5], hr, ((2,3), (1,)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((2,), (1,3,4,5,6)))
            TensorKit.mul!(cache[8], El, cache[7], 1, 0)
            permute!(cache[9], cache[8], ((6,2,1,4,5), (3,)))
            TensorKit.mul!(cache[10], cache[9], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[10], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,2}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_12_11_3_m" begin
        if isempty(cache)
            c1 = permute(hl, ((1,2), (3,)); copy=true)          # [d1,g; bL]
            c2 = permute(objA, ((2,), (1,3,4,5,6)); copy=true)  # [bL; bR,a,r,tR,tL]
            c3 = c1 * c2                                        # [d1,g; bR,a,r,tR,tL]
            c4 = permute(c3, ((3,), (1,2,4,5,6,7)); copy=true)  # [bR; d1,g,a,r,tR,tL]
            c5 = hr * c4                                        # [d2; d1,g,a,r,tR,tL]
            c6 = permute(c5, ((4,), (1,2,3,5,6,7)); copy=true)  # [a; d2,d1,g,r,tR,tL]
            c7 = El * c6                                        # [a'; d2,d1,g,r,tR,tL]
            c8 = permute(c7, ((2,3,1,6,7), (5,4)); copy=true)   # [d2,d1,a',tR,tL; r,g]
            c9 = c8 * Er                                        # [d2,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9])
        else
            permute!(cache[1], hl, ((1,2), (3,)))
            permute!(cache[2], objA, ((2,), (1,3,4,5,6)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((3,), (1,2,4,5,6,7)))
            TensorKit.mul!(cache[5], hr, cache[4], 1, 0)
            permute!(cache[6], cache[5], ((4,), (1,2,3,5,6,7)))
            TensorKit.mul!(cache[7], El, cache[6], 1, 0)
            permute!(cache[8], cache[7], ((2,3,1,6,7), (5,4)))
            TensorKit.mul!(cache[9], cache[8], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[9], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,2}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; Er = ErW.A
    @timeit to "_action2_2_2_12_00_3_m" begin
        if isempty(cache)
            c1 = permute(hl, ((1,2), (3,)); copy=true)          # [d1,g; bL]
            c2 = permute(objA, ((2,), (1,3,4,5,6)); copy=true)  # [bL; bR,a,r,tR,tL]
            c3 = c1 * c2                                        # [d1,g; bR,a,r,tR,tL]
            c4 = permute(c3, ((4,), (1,2,3,5,6,7)); copy=true)  # [a; d1,g,bR,r,tR,tL]
            c5 = El * c4                                        # [a'; d1,g,bR,r,tR,tL]
            c6 = permute(c5, ((4,2,1,6,7), (5,3)); copy=true)   # [bR,d1,a',tR,tL; r,g]
            c7 = c6 * Er                                        # [bR,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7])
        else
            permute!(cache[1], hl, ((1,2), (3,)))
            permute!(cache[2], objA, ((2,), (1,3,4,5,6)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((4,), (1,2,3,5,6,7)))
            TensorKit.mul!(cache[5], El, cache[4], 1, 0)
            permute!(cache[6], cache[5], ((4,2,1,6,7), (5,3)))
            TensorKit.mul!(cache[7], cache[6], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[7], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, hlW::LocalOperator{1,1}, hrW::LocalOperator{1,2}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_11_12_3_m" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3,4,5,6)); copy=true)  # [bL; bR,a,r,tR,tL]
            c2 = hl * c1                                        # [d1; bR,a,r,tR,tL]
            c3 = permute(c2, ((2,), (1,3,4,5,6)); copy=true)    # [bR; d1,a,r,tR,tL]
            c4 = permute(hr, ((1,2), (3,)); copy=true)          # [d2,g; bR]
            c5 = c4 * c3                                        # [d2,g; d1,a,r,tR,tL]
            c6 = permute(c5, ((4,), (1,2,3,5,6,7)); copy=true)  # [a; d2,g,d1,r,tR,tL]
            c7 = El * c6                                        # [a'; d2,g,d1,r,tR,tL]
            c8 = permute(c7, ((2,4,1,6,7), (5,3)); copy=true)   # [d2,d1,a',tR,tL; r,g]
            c9 = c8 * Er                                        # [d2,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9])
        else
            permute!(cache[1], objA, ((2,), (1,3,4,5,6)))
            TensorKit.mul!(cache[2], hl, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3,4,5,6)))
            permute!(cache[4], hr, ((1,2), (3,)))
            TensorKit.mul!(cache[5], cache[4], cache[3], 1, 0)
            permute!(cache[6], cache[5], ((4,), (1,2,3,5,6,7)))
            TensorKit.mul!(cache[7], El, cache[6], 1, 0)
            permute!(cache[8], cache[7], ((2,4,1,6,7), (5,3)))
            TensorKit.mul!(cache[9], cache[8], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[9], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, hrW::LocalOperator{1,2}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_2_00_12_3_m" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3,4,5,6)); copy=true)  # [bR; bL,a,r,tR,tL]
            c2 = permute(hr, ((1,2), (3,)); copy=true)          # [d2,g; bR]
            c3 = c2 * c1                                        # [d2,g; bL,a,r,tR,tL]
            c4 = permute(c3, ((4,), (1,2,3,5,6,7)); copy=true)  # [a; d2,g,bL,r,tR,tL]
            c5 = El * c4                                        # [a'; d2,g,bL,r,tR,tL]
            c6 = permute(c5, ((2,4,1,6,7), (5,3)); copy=true)   # [d2,bL,a',tR,tL; r,g]
            c7 = c6 * Er                                        # [d2,bL,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7])
        else
            permute!(cache[1], objA, ((1,), (2,3,4,5,6)))
            permute!(cache[2], hr, ((1,2), (3,)))
            TensorKit.mul!(cache[3], cache[2], cache[1], 1, 0)
            permute!(cache[4], cache[3], ((4,), (1,2,3,5,6,7)))
            TensorKit.mul!(cache[5], El, cache[4], 1, 0)
            permute!(cache[6], cache[5], ((2,4,1,6,7), (5,3)))
            TensorKit.mul!(cache[7], cache[6], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[7], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{3}, hlW::LocalOperator{2,1}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_3_21_11_2_m" begin
        if isempty(cache)
            c1 = permute(El, ((1,3), (2,)); copy=true)          # [a',a; β]
            c2 = permute(hl, ((2,), (1,3)); copy=true)          # [g; d1,bL]
            c3 = c1 * c2                                        # [a',a; d1,bL]
            c4 = permute(c3, ((1,3), (2,4)); copy=true)         # [a',d1; a,bL]
            c5 = permute(objA, ((3,2), (1,4,5,6)); copy=true)   # [a,bL; bR,r,tR,tL]
            c6 = c4 * c5                                        # [a',d1; bR,r,tR,tL]
            c7 = permute(c6, ((3,), (1,2,4,5,6)); copy=true)    # [bR; a',d1,r,tR,tL]
            c8 = hr * c7                                        # [d2; a',d1,r,tR,tL]
            c9 = permute(c8, ((1,3,2,5,6), (4,)); copy=true)    # [d2,d1,a',tR,tL; r]
            c10 = c9 * Er                                       # [d2,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])
        else
            permute!(cache[1], El, ((1,3), (2,)))
            permute!(cache[2], hl, ((2,), (1,3)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (2,4)))
            permute!(cache[5], objA, ((3,2), (1,4,5,6)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[8], hr, cache[7], 1, 0)
            permute!(cache[9], cache[8], ((1,3,2,5,6), (4,)))
            TensorKit.mul!(cache[10], cache[9], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[10], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{3}, hlW::LocalOperator{2,1}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; Er = ErW.A
    @timeit to "_action2_2_3_21_00_2_m" begin
        if isempty(cache)
            c1 = permute(El, ((1,3), (2,)); copy=true)          # [a',a; β]
            c2 = permute(hl, ((2,), (1,3)); copy=true)          # [g; d1,bL]
            c3 = c1 * c2                                        # [a',a; d1,bL]
            c4 = permute(c3, ((1,3), (2,4)); copy=true)         # [a',d1; a,bL]
            c5 = permute(objA, ((3,2), (1,4,5,6)); copy=true)   # [a,bL; bR,r,tR,tL]
            c6 = c4 * c5                                        # [a',d1; bR,r,tR,tL]
            c7 = permute(c6, ((3,2,1,5,6), (4,)); copy=true)    # [bR,d1,a',tR,tL; r]
            c8 = c7 * Er                                        # [bR,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8])
        else
            permute!(cache[1], El, ((1,3), (2,)))
            permute!(cache[2], hl, ((2,), (1,3)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (2,4)))
            permute!(cache[5], objA, ((3,2), (1,4,5,6)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((3,2,1,5,6), (4,)))
            TensorKit.mul!(cache[8], cache[7], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[8], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{3}, hlW::LocalOperator{1,1}, hrW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_3_11_11_3_m" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3,4,5,6)); copy=true)  # [bL; bR,a,r,tR,tL]
            c2 = hl * c1                                        # [d1; bR,a,r,tR,tL]
            c3 = permute(c2, ((2,), (1,3,4,5,6)); copy=true)    # [bR; d1,a,r,tR,tL]
            c4 = hr * c3                                        # [d2; d1,a,r,tR,tL]
            c5 = permute(c4, ((3,), (1,2,4,5,6)); copy=true)    # [a; d2,d1,r,tR,tL]
            c6 = permute(El, ((1,2), (3,)); copy=true)          # [a',β; a]
            c7 = c6 * c5                                        # [a',β; d2,d1,r,tR,tL]
            c8 = permute(c7, ((3,4,1,6,7), (5,2)); copy=true)   # [d2,d1,a',tR,tL; r,β]
            c9 = c8 * Er                                        # [d2,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9])
        else
            permute!(cache[1], objA, ((2,), (1,3,4,5,6)))
            TensorKit.mul!(cache[2], hl, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3,4,5,6)))
            TensorKit.mul!(cache[4], hr, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((3,), (1,2,4,5,6)))
            permute!(cache[6], El, ((1,2), (3,)))
            TensorKit.mul!(cache[7], cache[6], cache[5], 1, 0)
            permute!(cache[8], cache[7], ((3,4,1,6,7), (5,2)))
            TensorKit.mul!(cache[9], cache[8], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[9], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{3}, wm)
    objA = objW.A; El = ElW.A; Er = ErW.A
    @timeit to "_action2_2_3_00_00_3_m" begin
        if isempty(cache)
            c1 = permute(objA, ((3,), (1,2,4,5,6)); copy=true)  # [a; bR,bL,r,tR,tL]
            c2 = permute(El, ((1,2), (3,)); copy=true)          # [a',β; a]
            c3 = c2 * c1                                        # [a',β; bR,bL,r,tR,tL]
            c4 = permute(c3, ((3,4,1,6,7), (5,2)); copy=true)   # [bR,bL,a',tR,tL; r,β]
            c5 = c4 * Er                                        # [bR,bL,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5])
        else
            permute!(cache[1], objA, ((3,), (1,2,4,5,6)))
            permute!(cache[2], El, ((1,2), (3,)))
            TensorKit.mul!(cache[3], cache[2], cache[1], 1, 0)
            permute!(cache[4], cache[3], ((3,4,1,6,7), (5,2)))
            TensorKit.mul!(cache[5], cache[4], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[5], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{3}, hlW::LocalOperator{1,1}, hrW::LocalOperator{2,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hl = hlW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_3_11_21_2_m" begin
        if isempty(cache)
            c1 = permute(El, ((1,3), (2,)); copy=true)          # [a',a; β]
            c2 = permute(hr, ((2,), (1,3)); copy=true)          # [g; d2,bR]
            c3 = c1 * c2                                        # [a',a; d2,bR]
            c4 = permute(c3, ((1,3), (2,4)); copy=true)         # [a',d2; a,bR]
            c5 = permute(objA, ((3,1), (2,4,5,6)); copy=true)   # [a,bR; bL,r,tR,tL]
            c6 = c4 * c5                                        # [a',d2; bL,r,tR,tL]
            c7 = permute(c6, ((3,), (1,2,4,5,6)); copy=true)    # [bL; a',d2,r,tR,tL]
            c8 = hl * c7                                        # [d1; a',d2,r,tR,tL]
            c9 = permute(c8, ((3,1,2,5,6), (4,)); copy=true)    # [d2,d1,a',tR,tL; r]
            c10 = c9 * Er                                       # [d2,d1,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10])
        else
            permute!(cache[1], El, ((1,3), (2,)))
            permute!(cache[2], hr, ((2,), (1,3)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (2,4)))
            permute!(cache[5], objA, ((3,1), (2,4,5,6)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((3,), (1,2,4,5,6)))
            TensorKit.mul!(cache[8], hl, cache[7], 1, 0)
            permute!(cache[9], cache[8], ((3,1,2,5,6), (4,)))
            TensorKit.mul!(cache[10], cache[9], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[10], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end

function _action2_contract(to::TimerOutput, cache::Vector{Any}, acc::CompositeMPOTensor{2,6}, objW::CompositeMPOTensor{2,6}, ElW::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, hrW::LocalOperator{2,1}, ErW::RightEnvironmentTensor{2}, wm)
    objA = objW.A; El = ElW.A; hr = hrW.A; Er = ErW.A
    @timeit to "_action2_2_3_00_21_2_m" begin
        if isempty(cache)
            c1 = permute(El, ((1,3), (2,)); copy=true)          # [a',a; β]
            c2 = permute(hr, ((2,), (1,3)); copy=true)          # [g; d2,bR]
            c3 = c1 * c2                                        # [a',a; d2,bR]
            c4 = permute(c3, ((1,3), (2,4)); copy=true)         # [a',d2; a,bR]
            c5 = permute(objA, ((3,1), (2,4,5,6)); copy=true)   # [a,bR; bL,r,tR,tL]
            c6 = c4 * c5                                        # [a',d2; bL,r,tR,tL]
            c7 = permute(c6, ((2,3,1,5,6), (4,)); copy=true)    # [d2,bL,a',tR,tL; r]
            c8 = c7 * Er                                        # [d2,bL,a',tR,tL; f]
            empty!(cache); append!(cache, [c1,c2,c3,c4,c5,c6,c7,c8])
        else
            permute!(cache[1], El, ((1,3), (2,)))
            permute!(cache[2], hr, ((2,), (1,3)))
            TensorKit.mul!(cache[3], cache[1], cache[2], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (2,4)))
            permute!(cache[5], objA, ((3,1), (2,4,5,6)))
            TensorKit.mul!(cache[6], cache[4], cache[5], 1, 0)
            permute!(cache[7], cache[6], ((2,3,1,5,6), (4,)))
            TensorKit.mul!(cache[8], cache[7], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[8], ((1,2,3), (6,4,5)), wm, 1)
    end
    return acc
end
