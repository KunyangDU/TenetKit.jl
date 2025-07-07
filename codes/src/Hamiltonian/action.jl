
function action(O::SparseProjectiveHamiltonian{0}, obj::T) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    x = nothing
    to = get_timer("action")
    timer_acc = TimerOutput()
    Nthr = get_num_threads_julia()
    @timeit to "action" if Nthr > 1
        xs = Vector{Any}(nothing,Nthr)
        to_accs = [TimerOutput() for _ in 1:Nthr]
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                tid = Threads.threadid()
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action0(O,obj,O.validinds[ct])
                try
                    xs[tid] = axpy!(1,C,xs[tid])
                    merge!(to_accs[tid], localto)
                catch
                    rethrow()
                end
            end
        end
        x = sum(filter(x -> !isnothing(x),xs))
        map(1:Nthr) do x 
            merge!(to,to_accs[x];tree_point = ["action"])
        end
    else
        for ind in O.validinds
            C,localto = _action0(O,obj,ind)
            x = axpy!(1,C,x)
            merge!(timer_acc, localto)
        end
    end
    merge!(to,timer_acc;tree_point = ["action"])
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))

    return x
end

function action(O::SparseProjectiveHamiltonian{1}, obj::Union{MPSTensor{3},DenseMPOTensor{4}})
    x = nothing
    to = get_timer("action")
    timer_acc = TimerOutput()
    Nthr = get_num_threads_julia()
    @timeit to "action" if Nthr > 1
        xs = Vector{Any}(nothing,Nthr)
        to_accs = [TimerOutput() for _ in 1:Nthr]
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                tid = Threads.threadid()
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action1(O,obj,O.validinds[ct])
                try
                    xs[tid] = axpy!(1,C,xs[tid])
                    merge!(to_accs[tid], localto)
                catch
                    rethrow()
                end
            end
        end
        x = sum(filter(x -> !isnothing(x),xs))
        map(1:Nthr) do x 
            merge!(timer_acc,to_accs[x])
        end
    else
        for ind in O.validinds
            C,localto = _action1(O,obj,ind)
            x = axpy!(1,C,x)
            merge!(timer_acc, localto)
        end
    end
    merge!(to,timer_acc;tree_point = ["action"])
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))

    return x
end

function action(O::SparseProjectiveHamiltonian{2}, obj::Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}})
    x = nothing
    to = get_timer("action")
    timer_acc = TimerOutput()
    Nthr = get_num_threads_julia()
    @timeit to "action" if Nthr > 1
        xs = Vector{Any}(nothing,Nthr)
        to_accs = [TimerOutput() for _ in 1:Nthr]
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                tid = Threads.threadid()
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action2(O,obj,O.validinds[ct])
                try
                    xs[tid] = axpy!(1,C,xs[tid])
                    merge!(to_accs[tid], localto)
                catch
                    rethrow()
                end
            end
        end
        x = sum(filter(x -> !isnothing(x),xs))
        map(1:Nthr) do x 
            merge!(to,to_accs[x];tree_point = ["action"])
        end
    else
        for ind in O.validinds
            C,localto = _action2(O,obj,ind)
            x = axpy!(1,C,x)
            merge!(timer_acc, localto)
        end
    end
    merge!(to,timer_acc;tree_point = ["action"])
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))

    return x
end

function action(O::SparseProjectiveHamiltonian{2}, tl::T, tr::T) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    x = nothing
    to = get_timer("action")
    timer_acc = TimerOutput()
    Nthr = get_num_threads_julia()
    @timeit to "action" if Nthr > 1
        xs = Vector{Any}(nothing,Nthr)
        to_accs = [TimerOutput() for _ in 1:Nthr]
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                tid = Threads.threadid()
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action2(O,tl,tr,O.validinds[ct])
                try
                    xs[tid] = axpy!(1,C,xs[tid])
                    merge!(to_accs[tid], localto)
                catch
                    rethrow()
                end
            end
        end
        x = sum(filter(x -> !isnothing(x),xs))
        map(1:Nthr) do x 
            merge!(to,to_accs[x];tree_point = ["action"])
        end
    else
        for ind in O.validinds
            C,localto = _action2(O,tl,tr,ind)
            x = axpy!(1,C,x)
            merge!(timer_acc, localto)
        end
    end
    merge!(to,timer_acc;tree_point = ["action"])
    !iszero(O.E₀) && (x = axpy!(-O.E₀, obj, x))

    return x
end

function action(O::DenseProjectiveHamiltonian{2,1}, obj::DenseMPOTensor{4})
    @tensor x[-1 -2;-3 -4] ≔ O.EnvL.A.A[-2,1] * obj.A[-1,1,2,-4] * O.EnvR.A.A[2,-3]
    return DenseMPOTensor(x)
end

# dirty detail, threads free

function _action0(O::SparseProjectiveHamiltonian{0}, obj::T,i::Int64) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    localto = TimerOutput()
    @timeit localto "_action0_EL=El_obj" EL = contract(O.EnvL.A[i], obj)
    @timeit localto "_action0_C=EL_Er" C = T(contract(EL,O.EnvR.A[i]))
    return C, localto
end

function _action1(O::SparseProjectiveHamiltonian{1}, obj::Union{MPSTensor{3},DenseMPOTensor{4}}, ind::Tuple)
    i,j = ind
    localto = TimerOutput()
    @timeit localto "_action1_EL=El_obj_H" EL = contract(O.EnvL.A[i], obj, O.H.ts[1].m[i,j])
    @timeit localto "_action1_C=EL_Er" C = contract(EL,O.EnvR.A[j])
    return C, localto
end

function _action2(O::SparseProjectiveHamiltonian{2}, obj::Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}}, ind::Tuple)
    i,j,k = ind
    localto = TimerOutput()
    @timeit localto "_action2_EL1=El_obj_H1" EL1 = contract(O.EnvL.A[i], obj, O.H.ts[1].m[i,j])
    @timeit localto "_action2_EL2=EL1_H2" EL2 = contract(EL1, O.H.ts[2].m[j,k])
    @timeit localto "_action2_C=EL2_Er" C = contract(EL2, O.EnvR.A[k])
    return C, localto
end

function _action2(O::SparseProjectiveHamiltonian{2}, tl::T, tr::T, ind::Tuple) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    i,j,k = ind
    localto = TimerOutput()
    @timeit localto "_action2_EL=El_tl_H1" EL = contract(O.EnvL.A[i], tl, O.H.ts[1].m[i,j])
    @timeit localto "_action2_ER=tr_H2_Er" ER = contract(tr, O.H.ts[2].m[j,k],O.EnvR.A[k])
    @timeit localto "_action2_C=EL_ER" C = contract(EL, ER)
    return C, localto
end
