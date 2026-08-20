# ====================== 零分配 action 缓存（cache 途径）======================
# 缓存存包装对象（LeftEnvironmentTensor/LocalOperator/IdentityOperator/RightEnvironmentTensor），
# 热循环里按 N（编译期）派发到 _action1_contract / _action2_contract（见 mps/mpo sparse.jl 与 dense.jl）。
# 累加器零向量由输入张量直接指定（zerovector(obj, TT)）：cache 只服务自同态（输出空间 == 输入空间，
# 高频场景如 Lanczos/DMRG/TDVP/CBE/TaSK/tr），不再从 El/Er/H 反推输出空间。

mutable struct _ActionCache
    El::Vector{Any}      # LeftEnvironmentTensor{2/3} 包装
    Er::Vector{Any}      # RightEnvironmentTensor{2/3} 包装
    hlA::Vector{Any}     # 左算符（1-site 的唯一算符也放这；2-site 左）
    hrA::Vector{Any}     # 右算符（仅 2-site；1-site 为 nothing）
    wmid::Vector{Number} # 中间权重（仅 2-site；1-site 为 1.0）
    TT::Type
    caches::Vector{Any}  # 每 validind 一套中间缓冲
    acc::Vector{Any}     # 每 worker 一个累加器
end

function _init_action_cache!(O::SparseProjectiveHamiltonian{1}, obj)
    n = length(O.validinds)
    El = Vector{Any}(undef, n); Er = Vector{Any}(undef, n)
    hlA = Vector{Any}(undef, n); hrA = Vector{Any}(undef, n)
    wmid = Vector{Number}(undef, n)
    TTs = Type[scalartype(obj.A)]
    for (i, (l_inds, j, r_inds, wl, wr)) in enumerate(O.validinds)
        El[i] = _wsum(O.EnvL, l_inds, wl); Er[i] = _wsum(O.EnvR, r_inds, wr)
        h = O.H[1][j]
        h isa Union{IdentityOperator{1},LocalOperator{1,1},LocalOperator{1,2},LocalOperator{2,1}} || return nothing
        hlA[i] = h; hrA[i] = nothing; wmid[i] = 1.0
        push!(TTs, scalartype(El[i].A), scalartype(Er[i].A))
        h isa IdentityOperator{1} || push!(TTs, scalartype(h.A))
    end
    TT = reduce(promote_type, TTs)
    caches = [Any[] for _ in 1:n]
    acc = [zerovector(obj, TT) for _ in 1:Threads.nthreads()]   # wrapper 累加器（空间由输入张量直接指定）
    c = _ActionCache(El, Er, hlA, hrA, wmid, TT, caches, acc)
    O.cache = c
    return c
end

function _init_action_cache!(O::SparseProjectiveHamiltonian{2}, obj)
    n = length(O.validinds)
    El = Vector{Any}(undef, n); Er = Vector{Any}(undef, n)
    hlA = Vector{Any}(undef, n); hrA = Vector{Any}(undef, n)
    wmid = Vector{Number}(undef, n)
    TTs = Type[scalartype(obj.A)]
    for (i, (l_inds, (j,k), r_inds, wl, w_mid, wr)) in enumerate(O.validinds)
        El[i] = _wsum(O.EnvL, l_inds, wl); Er[i] = _wsum(O.EnvR, r_inds, wr)
        hl = O.H[1][j]; hr = O.H[2][k]
        hl isa Union{IdentityOperator{1},LocalOperator{1,1},LocalOperator{1,2},LocalOperator{2,1}} || return nothing
        hr isa Union{IdentityOperator{1},LocalOperator{1,1},LocalOperator{1,2},LocalOperator{2,1}} || return nothing
        hlA[i] = hl; hrA[i] = hr; wmid[i] = w_mid
        push!(TTs, scalartype(El[i].A), scalartype(Er[i].A), typeof(w_mid))
        hl isa IdentityOperator{1} || push!(TTs, scalartype(hl.A))
        hr isa IdentityOperator{1} || push!(TTs, scalartype(hr.A))
    end
    TT = reduce(promote_type, TTs)
    caches = [Any[] for _ in 1:n]
    acc = [zerovector(obj, TT) for _ in 1:Threads.nthreads()]   # wrapper 累加器（空间由输入张量直接指定）
    c = _ActionCache(El, Er, hlA, hrA, wmid, TT, caches, acc)
    O.cache = c
    return c
end

function action(O::SparseProjectiveHamiltonian{N}, obj::T) where {N, T <: Union{MPSTensor{3}, CompositeMPSTensor{2,4}, DenseMPOTensor{4}, CompositeMPOTensor{2,6}}}
    c = O.cache === nothing ? _init_action_cache!(O, obj) : O.cache
    c === nothing && return actionb(O, obj)
    to = get_timer("action")
    n = length(O.validinds)
    objA = obj.A * one(c.TT)                  # 裸张量：TT 提升 + 强制新拷贝（解耦持久 acc，防 obj 变 ComplexF64 后别名）
    objW = T(objA)                            # 包成 T，供链函数按 obj 类型分发（未来补 MPO 用 DenseMPOTensor）
    Nthr = Threads.nthreads()
    xs      = Vector{Any}(nothing, Nthr)      # 必须在函数作用域（@spawn 闭包捕获坑，不能塞进 if 块）
    tos     = [TimerOutput() for _ in 1:Nthr] # 每 worker 一个线程本地计时器（避免锁争用）
    counter = Threads.Atomic{Int64}(1)
    @timeit to "action" begin
    Threads.@sync for w in 1:Nthr
        Threads.@spawn begin
            accw = c.acc[w]
            to_w = tos[w]
            TensorKit.zerovector!(accw)
            while true
                ct = Threads.atomic_add!(counter, 1)
                ct > n && break
                if N == 1
                    _action1_contract(to_w, c.caches[ct], accw, objW, c.El[ct], c.hlA[ct], c.Er[ct])
                else
                    _action2_contract(to_w, c.caches[ct], accw, objW, c.El[ct], c.hlA[ct], c.hrA[ct], c.Er[ct], c.wmid[ct])
                end
            end
            xs[w] = accw
        end
    end
    x = xs[1]::T
    for w in 2:Nthr
        axpy!(1, xs[w], x)
    end
    !iszero(O.E₀) && axpby!(-O.E₀, objW, 1.0, x)
    end
    for w in 1:Nthr
        merge!(to, tos[w]; tree_point = ["action"])   # per-term 细分合并进全局 "action" 下
    end
    return x
end

# 0-site 边界（proj0 / projleft0 / projright0）：无零分配链（单个 El·obj·Er 缩并），直接走 bare。
function action(O::SparseProjectiveHamiltonian{0}, obj::T) where T <: Union{MPSTensor{2}, DenseMPOTensor{2}}
    return actionb(O, obj)
end

# ====================== 稠密 action 缓存 =======================

mutable struct _DenseActionCache
    TT::Type
    cache::Vector{Any}   # 单套中间缓冲（惰性分配，跨 action 调用复用）
    acc::Any             # 单个累加器（wrapper，与 obj 同类型）
end

function _init_proj_action_cache!(O::DenseProjectiveHamiltonian{N,L}, obj) where {N,L}
    TTs = Type[scalartype(obj.A), scalartype(O.EnvL.A.A), scalartype(O.EnvR.A.A)]
    TT = reduce(promote_type, TTs)
    acc = zerovector(obj, TT)
    c = _DenseActionCache(TT, Any[], acc)
    O.cache = c
    return c
end

function _init_dense_action_cache!(O::DenseProjectiveHamiltonian{N,L}, obj) where {N,L}
    TTs = Type[scalartype(obj.A), scalartype(O.EnvL.A.A), scalartype(O.EnvR.A.A)]
    O.H === nothing || (for h in O.H; push!(TTs, scalartype(h.A)); end)
    TT = reduce(promote_type, TTs)
    acc = zerovector(obj, TT)
    c = _DenseActionCache(TT, Any[], acc)
    O.cache = c
    return c
end
