# ====================== 1-site MPS 零分配 action（缓存 + 原地 mul!/permute! + β=1 融合累加）======================
# 每 validind 一套中间缓冲（惰性分配，跨 action 调用复用）；每 worker 一个累加器；末步 β=1 累加，不分配结果张量。
# 统一函数名 _action1_contract，按包装类型多重分发（动态派发）：El/Er 是 LeftEnvironmentTensor{2/3} / RightEnvironmentTensor{2/3}，
# 算符是 LocalOperator{1,1}/{1,2}/{2,1} 或 IdentityOperator{1}。方法内部取 .A 得到裸张量后走原地缩并链。
# 索引约定照搬 src/Hamiltonian/contract.jl 的 _action1_contract：
#   obj[c,e,h]（codom (c,e), dom (h)），结果 [a,d;f]（Id 情形 [a,e;f]）
# 缓存分配用 permute(...; copy=true)（分配版 permute 可能返回共享内存 view，会被后续原地写破坏）。
# @timeit 用字面量标签（编译期常量，TimerOutputs 可预声明 section），to 为每 worker 线程本地计时器，无锁争用。

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

# ====================== 1-site MPO 零分配 action（DenseMPOTensor{4}）======================
# 与 MPS 同构：每 validind 一套中间缓冲 + 每 worker 一个累加器。
# objA = DenseMPOTensor{4}.A = [下(p), 左(a); 右(r), 上(t)]（codom (p,a)，dom (r,t)）。
# 结果 x = [X, a'; f, t]，X = d（有算符）或 p（Id）——注意 phys 在前、键在后（与 MPS 的 (a',d) 相反）。
# 环境/算符约定与 MPS 一致：El{2}=[a';a]、El{3}=[a';β,a]、Er{2}=[r;f]、Er{3}=[r,γ;f]；
# h{1,1}=[d;p]、h{1,2}=[d;g,p]、h{2,1}=[d,g;p]、Id 透传。带 aux 的链末步 mul! + add_permute!。

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::LeftEnvironmentTensor{2}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{2})   # Id, El{2}, Er{2}
    objA = objW.A; El = ElW.A; Er = ErW.A
    @timeit to "_action1_1_2_00_2_m" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3,4)); copy=true)    # [a; p,r,t]
            c2 = El * c1                                      # [a'; p,r,t]
            c3 = permute(c2, ((2,1,4), (3,)); copy=true)      # [p,a',t; r]
            c4 = c3 * Er                                      # [p,a',t; f]
            empty!(cache); append!(cache, [c1, c2, c3, c4])
        else
            permute!(cache[1], objA, ((2,), (1,3,4)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,1,4), (3,)))
            TensorKit.mul!(cache[4], cache[3], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[4], ((1,2), (4,3)), 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::LeftEnvironmentTensor{3}, ::IdentityOperator{1}, ErW::RightEnvironmentTensor{3})   # Id, El{3}, Er{3}
    objA = objW.A; El = ElW.A; Er = ErW.A
    @timeit to "_action1_1_3_00_3_m" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3,4)); copy=true)    # [a; p,r,t]
            c2 = permute(El, ((1,2), (3,)); copy=true)        # [a',β; a]
            c3 = c2 * c1                                      # [a',β; p,r,t]
            c4 = permute(c3, ((1,3,5), (4,2)); copy=true)     # [a',p,t; r,β]
            c5 = c4 * Er                                      # [a',p,t; f]
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5])
        else
            permute!(cache[1], objA, ((2,), (1,3,4)))
            permute!(cache[2], El, ((1,2), (3,)))
            TensorKit.mul!(cache[3], cache[2], cache[1], 1, 0)
            permute!(cache[4], cache[3], ((1,3,5), (4,2)))
            TensorKit.mul!(cache[5], cache[4], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[5], ((2,1), (4,3)), 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::LeftEnvironmentTensor{2}, hW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{2})   # LocalOperator{1,1}, El{2}, Er{2}
    objA = objW.A; El = ElW.A; h = hW.A; Er = ErW.A
    @timeit to "_action1_1_2_11_2_m" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3,4)); copy=true)    # [p; a,r,t]
            c2 = h * c1                                       # [d; a,r,t]
            c3 = permute(c2, ((2,), (1,3,4)); copy=true)      # [a; d,r,t]
            c4 = El * c3                                      # [a'; d,r,t]
            c5 = permute(c4, ((2,1,4), (3,)); copy=true)      # [d,a',t; r]
            c6 = c5 * Er                                      # [d,a',t; f]
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6])
        else
            permute!(cache[1], objA, ((1,), (2,3,4)))
            TensorKit.mul!(cache[2], h, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3,4)))
            TensorKit.mul!(cache[4], El, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((2,1,4), (3,)))
            TensorKit.mul!(cache[6], cache[5], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[6], ((1,2), (4,3)), 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::LeftEnvironmentTensor{3}, hW::LocalOperator{1,1}, ErW::RightEnvironmentTensor{3})   # LocalOperator{1,1}, El{3}, Er{3}
    objA = objW.A; El = ElW.A; h = hW.A; Er = ErW.A
    @timeit to "_action1_1_3_11_3_m" begin
        if isempty(cache)
            c1 = permute(objA, ((1,), (2,3,4)); copy=true)    # [p; a,r,t]
            c2 = h * c1                                       # [d; a,r,t]
            c3 = permute(c2, ((2,), (1,3,4)); copy=true)      # [a; d,r,t]
            c4 = permute(El, ((1,2), (3,)); copy=true)        # [a',β; a]
            c5 = c4 * c3                                      # [a',β; d,r,t]
            c6 = permute(c5, ((1,3,5), (4,2)); copy=true)     # [a',d,t; r,β]
            c7 = c6 * Er                                      # [a',d,t; f]
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6, c7])
        else
            permute!(cache[1], objA, ((1,), (2,3,4)))
            TensorKit.mul!(cache[2], h, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3,4)))
            permute!(cache[4], El, ((1,2), (3,)))
            TensorKit.mul!(cache[5], cache[4], cache[3], 1, 0)
            permute!(cache[6], cache[5], ((1,3,5), (4,2)))
            TensorKit.mul!(cache[7], cache[6], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[7], ((2,1), (4,3)), 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::LeftEnvironmentTensor{2}, hW::LocalOperator{1,2}, ErW::RightEnvironmentTensor{3})   # LocalOperator{1,2}, El{2}, Er{3}
    objA = objW.A; El = ElW.A; h = hW.A; Er = ErW.A
    @timeit to "_action1_1_2_12_3_m" begin
        if isempty(cache)
            c1 = permute(objA, ((2,), (1,3,4)); copy=true)    # [a; p,r,t]
            c2 = El * c1                                      # [a'; p,r,t]
            c3 = permute(c2, ((2,), (1,3,4)); copy=true)      # [p; a',r,t]
            c4 = permute(h, ((1,2), (3,)); copy=true)         # [d,g; p]
            c5 = c4 * c3                                      # [d,g; a',r,t]
            c6 = permute(c5, ((3,1,5), (4,2)); copy=true)     # [a',d,t; r,g]
            c7 = c6 * Er                                      # [a',d,t; f]
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6, c7])
        else
            permute!(cache[1], objA, ((2,), (1,3,4)))
            TensorKit.mul!(cache[2], El, cache[1], 1, 0)
            permute!(cache[3], cache[2], ((2,), (1,3,4)))
            permute!(cache[4], h, ((1,2), (3,)))
            TensorKit.mul!(cache[5], cache[4], cache[3], 1, 0)
            permute!(cache[6], cache[5], ((3,1,5), (4,2)))
            TensorKit.mul!(cache[7], cache[6], Er, 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[7], ((2,1), (4,3)), 1, 1)
    end
    return acc
end

function _action1_contract(to::TimerOutput, cache::Vector{Any}, acc::DenseMPOTensor{4}, objW::DenseMPOTensor{4}, ElW::LeftEnvironmentTensor{3}, hW::LocalOperator{2,1}, ErW::RightEnvironmentTensor{2})   # LocalOperator{2,1}, El{3}, Er{2}
    objA = objW.A; El = ElW.A; h = hW.A; Er = ErW.A
    @timeit to "_action1_1_3_21_2_m" begin
        if isempty(cache)
            c1 = permute(objA, ((1,2,4), (3,)); copy=true)    # [p,a,t; r]
            c2 = c1 * Er                                      # [p,a,t; f]
            c3 = permute(c2, ((1,), (2,3,4)); copy=true)      # [p; a,t,f]
            c4 = h * c3                                       # [d,g; a,t,f]
            c5 = permute(c4, ((2,3), (1,4,5)); copy=true)     # [g,a; d,t,f]
            c6 = El * c5                                      # [a'; d,t,f]
            empty!(cache); append!(cache, [c1, c2, c3, c4, c5, c6])
        else
            permute!(cache[1], objA, ((1,2,4), (3,)))
            TensorKit.mul!(cache[2], cache[1], Er, 1, 0)
            permute!(cache[3], cache[2], ((1,), (2,3,4)))
            TensorKit.mul!(cache[4], h, cache[3], 1, 0)
            permute!(cache[5], cache[4], ((2,3), (1,4,5)))
            TensorKit.mul!(cache[6], El, cache[5], 1, 0)
        end
        TensorKit.add_permute!(acc.A, cache[6], ((2,1), (4,3)), 1, 1)
    end
    return acc
end
