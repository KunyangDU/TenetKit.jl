# ====================== 1-site MPS 零分配 action 链（cache）======================
# 每 validind 一套中间缓冲（惰性分配，跨 action 调用复用）；每 worker 一个累加器；末步 β=1 累加，不分配结果张量。
# 统一函数名 _action1_contract，按包装类型多重分发（动态派发）：El/Er 是 LeftEnvironmentTensor{2/3} / RightEnvironmentTensor{2/3}，
# 算符是 LocalOperator{1,1}/{1,2}/{2,1} 或 IdentityOperator{1}。索引约定照搬 contract.jl 的 _action1_contract（@tensor 参考）。
function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::MPSTensor{3}, objW::MPSTensor{3}, ElW::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{2})   # Id, El{2}, Er{2}
    objA = objW.A; El = ElW.A; Er = ErW.A
    @timeit to "_action1_1_2_00_2" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3)); copy=true)
            c2 = El * c1
            c3 = permute(c2, ((1,2), (3,)); copy=true)
            empty!(cache); append!(cache, [c1, c2, c3])
        else
            permute!(cache[1], objA, ((1,), (2,3)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((1,2), (3,)))
        end
        TensorKit.mul!(acc.A, cache[3], Er, 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::MPSTensor{3}, objW::MPSTensor{3}, ElW::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{3})   # Id, El{3}, Er{3}
    objA = objW.A; El = ElW.A; Er = ErW.A
    @timeit to "_action1_1_3_00_3" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3)); copy=true)
            c2 = permute(El, ((1,2), (3,)); copy=true)
            c3 = c2 * c1
            c4 = permute(c3, ((1,3), (4,2)); copy=true)
            empty!(cache); append!(cache, [c1, c2, c3, c4])
        else
            permute!(cache[1], objA, ((1,), (2,3)))
            TensorKit.mul!(cache[3], cache[2], cache[1], 1, 0)
            permute!(cache[4], cache[3], ((1,3), (4,2)))
        end
        TensorKit.mul!(acc.A, cache[4], Er, 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::MPSTensor{3}, objW::MPSTensor{3}, ElW::LeftEnvironmentTensor{2}, hW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{2})   # LocalOperator{1,1}, El{2}, Er{2}
    objA = objW.A; El = ElW.A; h = hW.A; Er = ErW.A
    @timeit to "_action1_1_2_11_2" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3)); copy=true)
            c2 = h * c1
            c3 = permute(c2, ((2,), (1,3)); copy=true)
            c4 = El * c3
            c5 = permute(c4, ((1,2), (3,)); copy=true)
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5])
        else
            permute!(cache[1], objA, ((2,), (1,3)))
            TensorKit.mul!(cache[2], h, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3)))
            TensorKit.mul!(cache[4], El, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((1,2), (3,)))
        end
        TensorKit.mul!(acc.A, cache[5], Er, 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::MPSTensor{3}, objW::MPSTensor{3}, ElW::LeftEnvironmentTensor{3}, hW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{3})   # LocalOperator{1,1}, El{3}, Er{3}
    objA = objW.A; El = ElW.A; h = hW.A; Er = ErW.A
    @timeit to "_action1_1_3_11_3" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3)); copy=true)
            c2 = h * c1
            c3 = permute(c2, ((2,), (1,3)); copy=true)
            c4 = permute(El, ((1,2), (3,)); copy=true)
            c5 = c4 * c3
            c6 = permute(c5, ((1,3), (4,2)); copy=true)
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6])
        else
            permute!(cache[1], objA, ((2,), (1,3)))
            TensorKit.mul!(cache[2], h, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3)))
            TensorKit.mul!(cache[5], cache[4], cache[3], 1, 0)
            permute!(cache[6], cache[5], ((1,3), (4,2)))
        end
        TensorKit.mul!(acc.A, cache[6], Er, 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::MPSTensor{3}, objW::MPSTensor{3}, ElW::LeftEnvironmentTensor{2}, hW::LocalOperator{1,2}, ErW::RightEnvironmentTensor{3})   # LocalOperator{1,2}, El{2}, Er{3}；h 的 domain 是 (g,e)
    objA = objW.A; El = ElW.A; h = hW.A; Er = ErW.A
    @timeit to "_action1_1_2_12_3" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3)); copy=true)    # [c; e,h]
            c2 = El * c1                                     # [a; e,h]
            c3 = permute(c2, ((2,), (1,3)); copy=true)       # [e; a,h]
            c4 = permute(h, ((1,2), (3,)); copy=true)        # [d,g; e]（h domain (g,e) 合并 (d,g)）
            c5 = c4 * c3                                     # [d,g; a,h]
            c6 = permute(c5, ((3,1), (4,2)); copy=true)      # [a,d; h,g]
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6])
        else
            permute!(cache[1], objA, ((1,), (2,3)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3)))
            TensorKit.mul!(cache[5], cache[4], cache[3], 1, 0)
            permute!(cache[6], cache[5], ((3,1), (4,2)))
        end
        TensorKit.mul!(acc.A, cache[6], Er, 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::MPSTensor{3}, objW::MPSTensor{3}, ElW::LeftEnvironmentTensor{3}, hW::LocalOperator{2,1}, ErW::RightEnvironmentTensor{2})   # LocalOperator{2,1}, El{3}, Er{2}
    objA = objW.A; El = ElW.A; h = hW.A; Er = ErW.A
    @timeit to "_action1_1_3_21_2" begin
        if isempty(cache)
            c1 = objA * Er
            c2 = permute(c1, ((2,), (1,3)); copy=true)
            c3 = h * c2
            c4 = permute(c3, ((2,3), (1,4)); copy=true)
            c5 = El * c4
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5])
        else
            TensorKit.mul!(cache[1], objA, Er, 1, 0)
            permute!(cache[2], cache[1], ((2,), (1,3)))
            TensorKit.mul!(cache[3], h, cache[2], 1, 0)
            permute!(cache[4], cache[3], ((2,3), (1,4)))
            TensorKit.mul!(cache[5], El, cache[4], 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[5], ((1,2), (3,)), 1, 1)
    end
    return acc
end

# ====================== 2-site MPS 零分配 action 链（cache）======================

# 与 1-site 同构：统一函数名 _action2_contract，按包装类型多重分发。obj[a,p1,p2,r]，结果 [a',d1',d2';f]。

# 算符布局：{1,1}=[d;p]，{1,2}=[d;g,p]，{2,1}=[d,g;p]；环境 El{2}=[a';a]，El{3}=[a';b,a]，Er{2}=[r;f]，Er{3}=[r,b;f]。
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
