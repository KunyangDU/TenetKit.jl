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

function _action(O::SparseProjectiveHamiltonian{0}, obj::T, ind::Tuple{Vector{Int64},Nothing,Vector{Int64},Vector{Float64},Vector{Float64}}) where T <: Union{MPSTensor{2},DenseMPOTensor{2}}
    l_inds, ~, r_inds, wl, wr = ind
    tmp,localto = _action0(obj, _wsum(O.EnvL, l_inds, wl), _wsum(O.EnvR, r_inds, wr))
    return tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{1}, obj::T, ind::Tuple{Vector{Int64},Int64,Vector{Int64},Vector{Float64},Vector{Float64}}) where T <: Union{MPSTensor{3},DenseMPOTensor{4}}
    l_inds, j, r_inds, wl, wr = ind
    tmp,localto = _action1(obj, _wsum(O.EnvL, l_inds, wl), O.H[1][j], _wsum(O.EnvR, r_inds, wr))
    return tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{2}, obj::T, ind::Tuple{Vector{Int64},Tuple{Int64,Int64},Vector{Int64},Vector{Float64},Float64,Vector{Float64}}) where T <: Union{CompositeMPSTensor{2,4}, CompositeMPOTensor{2, 6}}
    l_inds, (j,k), r_inds, wl, w_mid, wr = ind
    tmp,localto = _action2(obj, _wsum(O.EnvL, l_inds, wl), O.H[1][j], O.H[2][k], _wsum(O.EnvR, r_inds, wr))
    return w_mid * tmp, localto
end

function _action(O::SparseProjectiveHamiltonian{2}, obj::SparseMPO{2}, ind::Tuple{Vector{Int64},Tuple{Int64,Int64},Vector{Int64},Vector{Float64},Float64,Vector{Float64}})
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

function action(O::DenseProjectiveHamiltonian{3, 2}, obj::CompositeMPOTensor{2, 6})
    to = get_timer("action")
    h1 = O.H[1].A
    h2 = O.H[2].A
    @timeit to "_action2_2_2_11_11_2" @tensor tmp[-1,-2,-3,-6,-7;-5,-4] ≔ O.EnvL.A.A[-1,1,2] * h1[-2,1,4,3] * h2[-3,4,-4,5] * obj.A[5,3,2,-5,-6,-7]
    return CompositeMPOTensor(permute(tmp * O.EnvR.A.A,(3,2,1),(6,4,5)))
end

function action(O::DenseProjectiveHamiltonian{3, 1}, obj::DenseMPOTensor{4})
    to = get_timer("action")
    h = O.H[1].A
    @timeit to "_action1_1_1_11_11_1" @tensor tmp[-1,-2,-5;-4,-3] ≔ O.EnvL.A.A[-1,1,2] * h[-2,1,-3,3] * obj.A[3,2,-4,-5]
    return DenseMPOTensor(permute(tmp * O.EnvR.A.A,(2,1),(4,3)))
end

