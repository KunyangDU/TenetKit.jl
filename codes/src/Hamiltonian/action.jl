function action(O::SparseProjectiveHamiltonian, obj::T) where T <: Union{MPSTensor{2},DenseMPOTensor{2},MPSTensor{3},DenseMPOTensor{4},CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}}
    x = nothing
    to = get_timer("action")
    timer_acc = TimerOutput()
    Nthr = get_nworker()
    @timeit to "action" if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action(O,obj,O.validinds[ct])
                lock(Lock)
                try
                    x = axpy!(1,C,x)
                    merge!(timer_acc, localto)
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for ind in O.validinds
            C,localto = _action(O,obj,ind)
            x = axpy!(1,C,x)
            merge!(timer_acc, localto)
        end
    end
    merge!(to,timer_acc;tree_point = ["action"])
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))

    return x
end

# ====================== 零分配 action（1-site 与 2-site 共用）======================
# 缓存存包装对象（LeftEnvironmentTensor/LocalOperator/IdentityOperator/RightEnvironmentTensor），
# 热循环里按 N（编译期）派发到 _action1_contract / _action2_contract（见 action1.jl / action2.jl）。
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

# ====================== 输出零张量构造（空间由 El/Er + obj 透传腿决定）======================
# 不能 similar(obj)：mul! 等一次性、上下不对称的网络里，输出键维 = A·B 组合键维（由 El/Er 决定），
# 与 obj 的键维不同；phys 腿与 obj 一致（算符不改变 phys 空间），MPO 的「上」腿从 obj 透传。
_action_output_zerovector(obj::MPSTensor{3}, El::LeftEnvironmentTensor, Er::RightEnvironmentTensor, TT::Type) =
    MPSTensor(TensorKit.zerovector(zeros(codomain(El.A)[1] ⊗ codomain(obj.A)[2], domain(Er.A)[1]), TT))

_action_output_zerovector(obj::DenseMPOTensor{4}, El::LeftEnvironmentTensor, Er::RightEnvironmentTensor, TT::Type) =
    DenseMPOTensor(TensorKit.zerovector(zeros(codomain(obj.A)[1] ⊗ codomain(El.A)[1], domain(Er.A)[1] ⊗ domain(obj.A)[2]), TT))

_action_output_zerovector(obj::CompositeMPSTensor{2,4}, El::LeftEnvironmentTensor, Er::RightEnvironmentTensor, TT::Type) =
    CompositeMPSTensor(TensorKit.zerovector(zeros(codomain(El.A)[1] ⊗ codomain(obj.A)[2] ⊗ codomain(obj.A)[3], domain(Er.A)[1]), TT))

_action_output_zerovector(obj::CompositeMPOTensor{2,6}, El::LeftEnvironmentTensor, Er::RightEnvironmentTensor, TT::Type) =
    CompositeMPOTensor(TensorKit.zerovector(zeros(codomain(obj.A)[1] ⊗ codomain(obj.A)[2] ⊗ codomain(El.A)[1], domain(Er.A)[1] ⊗ domain(obj.A)[2] ⊗ domain(obj.A)[3]), TT))

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
    acc = [_action_output_zerovector(obj, El[1], Er[1], TT) for _ in 1:Threads.nthreads()]   # wrapper 累加器（空间由 El/Er + obj 透传腿决定）
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
    acc = [_action_output_zerovector(obj, El[1], Er[1], TT) for _ in 1:Threads.nthreads()]   # wrapper 累加器（空间由 El/Er + obj 透传腿决定）
    c = _ActionCache(El, Er, hlA, hrA, wmid, TT, caches, acc)
    O.cache = c
    return c
end

function action(O::SparseProjectiveHamiltonian{N}, obj::T) where {N, T <: Union{MPSTensor{3}, CompositeMPSTensor{2,4}, DenseMPOTensor{4}, CompositeMPOTensor{2,6}}}
    c = O.cache === nothing ? _init_action_cache!(O, obj) : O.cache
    c === nothing && return invoke(action, Tuple{SparseProjectiveHamiltonian, T}, O, obj)
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

function action(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2})
    x = nothing
    to = get_timer("action")
    timer_acc = TimerOutput()
    Nthr = get_nworker()
    for ind in O.validinds
        C,localto = _action(O,obj,ind)
        x = axpy!(1,C,x)
        merge!(timer_acc, localto)
    end
    return x
end

# dirty detail, threads free

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

