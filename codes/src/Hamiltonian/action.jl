
function action(O::SparseProjectiveHamiltonian{0}, obj::T) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    x = nothing
    to = get_timer("action")
    timer_acc = TimerOutput()
    Nthr = get_num_threads_julia()
    @timeit to "action" if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action0(O,obj,O.validinds[ct])
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
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action1(O,obj,O.validinds[ct])
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
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action2(O,obj,O.validinds[ct])
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
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(O.validinds) && break
                C,localto = _action2(O,tl,tr,O.validinds[ct])
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

function action(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2})
    x = nothing
    to = get_timer("action")
    timer_acc = TimerOutput()
    Nthr = get_num_threads_julia()
    for ind in O.validinds
        C,localto = _action2(O,obj,ind)
        x = axpy!(1,C,x)
        merge!(timer_acc, localto)
    end
    return x
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

# function _action2(O::SparseProjectiveHamiltonian{2}, obj::T, ind::Tuple) where T <: Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}}
#     i,j,k = ind
#     tmp,localto = _action2(obj,O.EnvL.A[i],O.H.ts[1].m[i,j],O.H.ts[2].m[j,k],O.EnvR.A[k])
#     return T(tmp), localto
# end

# function _action2(obj::Union{CompositeMPOTensor{2,6},CompositeMPSTensor{2,4}},El::LeftEnvironmentTensor{el},hl::AbstractLocalOperator{hl1,hl2},hr::AbstractLocalOperator{hr1,hr2},Er::RightEnvironmentTensor{er}) where {el,hl1,hl2,hr1,hr2,er}
#     localto = TimerOutput()
#     @timeit localto "_action2_2_$(el)_$(hl1)$(hl2)_$(hr1)$(hr2)_$(er)" tmp = _action2_contract(obj,El,hl,hr,Er)
#     return tmp, localto
# end

# function _action2_contract(obj::CompositeMPSTensor{2,4},El::LeftEnvironmentTensor{2},hl::LocalOperator{1,1},hr::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
#     return @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,2,3,4] * hl.A[-2,2] * hr.A[-3,3] * Er.A[4,-4]
# end
# function _action2_contract(obj::CompositeMPSTensor{2,4},El::LeftEnvironmentTensor{2},hl::IdentityOperator{1},hr::LocalOperator{1,1},Er::RightEnvironmentTensor{2})
#     return @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,-2,3,4] * hr.A[-3,3] * Er.A[4,-4]
# end
# function _action2_contract(obj::CompositeMPSTensor{2,4},El::LeftEnvironmentTensor{2},hl::LocalOperator{1,1},hr::IdentityOperator{1},Er::RightEnvironmentTensor{2})
#     return @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,2,-3,4] * hl.A[-2,2] * Er.A[4,-4]
# end
# function _action2_contract(obj::CompositeMPSTensor{2,4},El::LeftEnvironmentTensor{2},hl::IdentityOperator{1},hr::IdentityOperator{1},Er::RightEnvironmentTensor{2})
#     return @tensor tmp[-1,-2,-3;-4] ≔ El.A[-1,1] * obj.A[1,-2,-3,4] * Er.A[4,-4]
# end


function _action2(O::SparseProjectiveHamiltonian{2}, tl::T, tr::T, ind::Tuple) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    i,j,k = ind
    localto = TimerOutput()
    @timeit localto "_action2_EL=El_tl_H1" EL = contract(O.EnvL.A[i], tl, O.H.ts[1].m[i,j])
    @timeit localto "_action2_ER=tr_H2_Er" ER = contract(tr, O.H.ts[2].m[j,k],O.EnvR.A[k])
    @timeit localto "_action2_C=EL_ER" C = contract(EL, ER)
    return C, localto
end

function _action2(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2}, ind::Tuple)
    i,j,k = ind
    localto = TimerOutput()
    @timeit localto "_action2_EL1=El_H1" EL1 = contract(O.EnvL.A[i], obj.ts[1].m[i,j])
    @timeit localto "_action2_EL2=EL1_H2" EL2 = contract(EL1, obj.ts[2].m[j,k])
    @timeit localto "_action2_C=EL2_Er" C = contract(EL2, O.EnvR.A[k])
    return C, localto
end
