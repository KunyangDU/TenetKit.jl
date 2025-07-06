
function action(O::SparseProjectiveHamiltonian{0}, obj::T) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    x = nothing
    to = get_timer("action")
    Nthr = get_num_threads_julia()
    @timeit to "action" if Nthr > 1
        xs = Vector{Any}(nothing,Nthr)
        to_accs = [TimerOutput() for _ in 1:Nthr]
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                id = Threads.threadid()
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                localto = TimerOutput()
                @timeit localto "_action0" C = _action0(O,obj,O.validinds[ct])
                try
                    xs[id] = axpy!(1,C,xs[id])
                    merge!(to_accs[id], localto)
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
            C = _action0(O,obj,ind)
            x = axpy!(1,C,x)
        end
    end
    return x
end

function action(O::SparseProjectiveHamiltonian{1}, obj::Union{MPSTensor{3},DenseMPOTensor{4}})
    x = nothing
    to = get_timer("action")
    Nthr = get_num_threads_julia()
    @timeit to "action" if Nthr > 1
        xs = Vector{Any}(nothing,Nthr)
        to_accs = [TimerOutput() for _ in 1:Nthr]
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                id = Threads.threadid()
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                localto = TimerOutput()
                @timeit localto "_action1" C = _action1(O,obj,O.validinds[ct])
                try
                    xs[id] = axpy!(1,C,xs[id])
                    merge!(to_accs[id], localto)
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
            @timeit localto "_action1" C = _action1(O,obj,ind)
            x = axpy!(1,C,x)
        end
    end

    return x
end

function action(O::SparseProjectiveHamiltonian{2}, obj::Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}})
    x = nothing
    to = get_timer("action")
    Nthr = get_num_threads_julia()
    @timeit to "action" if Nthr > 1
        xs = Vector{Any}(nothing,Nthr)
        to_accs = [TimerOutput() for _ in 1:Nthr]
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Threads.nthreads()
            Threads.@spawn while true
                id = Threads.threadid()
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                localto = TimerOutput()
                @timeit localto "_action2" C = _action2(O,obj,O.validinds[ct])
                try
                    xs[id] = axpy!(1,C,xs[id])
                    merge!(to_accs[id], localto)
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
            @timeit localto "_action2" C = _action2(O,obj,ind)
            x = axpy!(1,C,x)
        end
    end

    return x
end

function action(O::SparseProjectiveHamiltonian{2}, tl::T, tr::T) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    x = nothing
    for ind in O.validinds
        C = _action2(O,tl,tr,ind)
        x = axpy!(1,C,x)
    end

    return x
end

function action(O::DenseProjectiveHamiltonian{2,1}, obj::DenseMPOTensor{4})
    @tensor x[-1 -2;-3 -4] ≔ O.EnvL.A.A[-2,1] * obj.A[-1,1,2,-4] * O.EnvR.A.A[2,-3]
    return DenseMPOTensor(x)
end

# dirty detail, threads free

function _action0(O::SparseProjectiveHamiltonian{0}, obj::T,i::Int64) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    EL = contract(O.EnvL.A[i], obj)
    return T(contract(EL,O.EnvR.A[i]))
end

function _action1(O::SparseProjectiveHamiltonian{1}, obj::Union{MPSTensor{3},DenseMPOTensor{4}}, ind::Tuple)
    i,j = ind
    EL = contract(O.EnvL.A[i], obj, O.H.ts[1].m[i,j])
    return contract(EL,O.EnvR.A[j])
end

function _action2(O::SparseProjectiveHamiltonian{2}, obj::Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}}, ind::Tuple)
    i,j,k = ind
    EL = contract(contract(O.EnvL.A[i], obj, O.H.ts[1].m[i,j]), O.H.ts[2].m[j,k])
    return contract(EL, O.EnvR.A[k])
end

function _action2(O::SparseProjectiveHamiltonian{2}, tl::T, tr::T, ind::Tuple) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    i,j,k = ind
    EL = contract(O.EnvL.A[i], tl, O.H.ts[1].m[i,j])
    ER = contract(tr, O.H.ts[2].m[j,k],O.EnvR.A[k])
    return contract(EL, ER)
end
