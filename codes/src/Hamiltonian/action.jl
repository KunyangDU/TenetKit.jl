function action(O::SparseProjectiveHamiltonian, obj::T) where T <: Union{MPSTensor{2},DenseMPOTensor{2},MPSTensor{3},DenseMPOTensor{4},CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}}
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
        C,localto = _action(O,obj,ind)
        x = axpy!(1,C,x)
        merge!(timer_acc, localto)
    end
    return x
end

# dirty detail, threads free

function _action(O::SparseProjectiveHamiltonian{0}, obj::T,i::Int64) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    tmp,localto = _action0(obj,O.EnvL.A[i],O.EnvR.A[i])
    return tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{1}, obj::T, ind::Tuple) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    i,j = ind
    tmp,localto = _action1(obj,O.EnvL.A[i],O.H.ts[1].m[i,j],O.EnvR.A[j])
    return tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{2}, obj::T, ind::Tuple) where T <: Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}}
    i,j,k = ind
    tmp,localto = _action2(obj,O.EnvL.A[i],O.H.ts[1].m[i,j],O.H.ts[2].m[j,k],O.EnvR.A[k])
    return tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2}, ind::Tuple)
    i,j,k = ind
    localto = TimerOutput()
    @timeit localto "_action2_EL1=El_H1" EL1 = contract(O.EnvL.A[i], obj.ts[1].m[i,j])
    @timeit localto "_action2_EL2=EL1_H2" EL2 = contract(EL1, obj.ts[2].m[j,k])
    @timeit localto "_action2_C=EL2_Er" C = contract(EL2, O.EnvR.A[k])
    return C, localto
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

