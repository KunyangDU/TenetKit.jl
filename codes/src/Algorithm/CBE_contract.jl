
"""
CBE
"""
function contract(El::LeftEnvironmentTensor{<:Number, 2},A::MPSTensor{<:Number, 3}, ::IdentityOperator{1})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,1] * A.A[1,-2,-3]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{<:Number, 2},A::MPSTensor{<:Number, 3}, mpo::LocalOperator{1,1})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,1] * A.A[1,2,-3] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{<:Number, 2},A::MPSTensor{<:Number, 3}, mpo::LocalOperator{1,2})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[1,2,-4] * mpo.A[-2,-3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{<:Number, 3},A::MPSTensor{<:Number, 3}, ::IdentityOperator{1})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,-3,1] * A.A[1,-2,-4]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{<:Number, 3},A::MPSTensor{<:Number, 3}, mpo::LocalOperator{2,1})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,3,1] * A.A[1,2,-3] * mpo.A[-2,3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{<:Number, 3}, B::LocalOperator{1,1}, EnvR::RightEnvironmentTensor{<:Number, 2})
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,2,1] * B.A[-2,2] * EnvR.A[1,-3]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::MPSTensor{<:Number, 3}, ::IdentityOperator{1}, EnvR::RightEnvironmentTensor{<:Number, 2})
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,-2,1] * EnvR.A[1,-3]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::MPSTensor{<:Number, 3}, B::LocalOperator{1,2}, EnvR::RightEnvironmentTensor{<:Number, 3})
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,3,1] * B.A[-2,2,3] * EnvR.A[1,2,-3]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::MPSTensor{<:Number, 3}, ::IdentityOperator{1}, EnvR::RightEnvironmentTensor{<:Number, 3})
    @tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,-3,1] * EnvR.A[1,-2,-4]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::MPSTensor{<:Number, 3}, B::LocalOperator{2,1}, EnvR::RightEnvironmentTensor{<:Number, 2})
    @tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,2,1] * B.A[-3,-2,2] * EnvR.A[1,-4]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::SparseLeftEnvironmentTensor{1},EnvR::SparseRightEnvironmentTensor{1})
    @assert (w = EnvL.D[1]) == EnvR.D[1]
    mps = nothing 
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > w && break
                C = contract(EnvL.A[ct],EnvR.A[ct])
                lock(Lock)
                try
                    mps = axpy!(1, C, mps)
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for i in 1:w 
            mps = axpy!(1,contract(EnvL.A[i],EnvR.A[i]),mps)
        end
    end
    return mps
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 3}, A::AdjointMPSTensor{<:Number, 3})
    @tensor tmp[-1;-2] ≔ EnvL.A[1,2,-2] * A.A[-1,1,2] 
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 1, 3}, A::AdjointMPSTensor{<:Number, 3})
    @tensor tmp[-1;-2] ≔ EnvR.A[-1,2,1] * A.A[1,-2,2] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, EnvR::RightCompositeEnvironmentTensor{<:Number, 1, 3})
    @tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,1] * EnvR.A[1,-2,-3]
    return MPSTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2,3}, A::MPSTensor{<:Number, 3})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3])
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 1,3}, B::MPSTensor{<:Number, 3})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,2,1] * B'.A[1,3,2] * B.A[3,-2,-3])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 3}, Λ::MPSTensor{<:Number, 2})
    return LeftCompositeEnvironmentTensor(EnvL.A*Λ.A)
end

function contract(EnvL::RightCompositeEnvironmentTensor{<:Number, 1, 3}, Λ::MPSTensor{<:Number, 2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2,4}, A::MPSTensor{<:Number, 3})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,-4] * A'.A[3,1,2] * A.A[-1,-2,3])
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 1,4}, B::MPSTensor{<:Number, 3})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4] ≔ EnvR.A[-1,-2,2,1] * B'.A[1,3,2] * B.A[3,-3,-4])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, Λ::MPSTensor{<:Number, 2})
    return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-1,-2,-3,1]*Λ.A[1,-4])
end

function contract(EnvL::RightCompositeEnvironmentTensor{<:Number, 1, 4}, Λ::MPSTensor{<:Number, 2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3,-4])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, A::AdjointMPSTensor{<:Number, 3})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[1,2,-2,-3] * A.A[-1,1,2] 
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 1, 4}, A::AdjointMPSTensor{<:Number, 3})
    @tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,-2,2,1] * A.A[1,-3,2] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, EnvR::RightCompositeEnvironmentTensor{<:Number, 1, 4})
    @tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,2,1] * EnvR.A[1,2,-2,-3]
    return MPSTensor(tmp)
end

function contract(El::LeftCompositeEnvironmentTensor{<:Number, 2,3},Er::RightEnvironmentTensor{<:Number, 2})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,1] * Er.A[1,-3]
    return MPSTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{<:Number, 2,4},Er::RightEnvironmentTensor{<:Number, 3})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,2,1] * Er.A[1,2,-3]
    return MPSTensor(tmp)
end

function contract(El::LeftEnvironmentTensor{<:Number, 3}, A::MPSTensor{<:Number, 3}, mpo::LocalOperator{1,1})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,-3,1] * A.A[1,2,-4] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(A::MPSTensor{<:Number, 3}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{<:Number, 3})
    @tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,2,1] * B.A[-3,2] * Er.A[1,-2,-4]
    return RightCompositeEnvironmentTensor(tmp)
end

"""
CBE
"""
function contract(El::LeftEnvironmentTensor{<:Number, 2},A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[2,1,-3,-4] * B.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{<:Number, 2},A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[-2,1,-3,-4]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{<:Number, 2},A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,2})
    @tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,1] * A.A[2,1,-4,-5] * B.A[-2,-3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{2,1})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-1,2,1] * A.A[3,1,-3,-4] * B.A[-2,2,3]
    return LeftCompositeEnvironmentTensor(tmp)
end


function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{<:Number, 3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,2] * Er.A[1,-2,-4]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1},Er::RightEnvironmentTensor{<:Number, 3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[-3,-1,1,-5] * Er.A[1,-2,-4]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,1},Er::RightEnvironmentTensor{<:Number, 2})
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[2,-1,1,-4] * B.A[-2,2] * Er.A[1,-3]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{<:Number, 4}, ::IdentityOperator{1},Er::RightEnvironmentTensor{<:Number, 2})
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[-2,-1,1,-4] * Er.A[1,-3]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{2,1},Er::RightEnvironmentTensor{<:Number, 2})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,-2,2] * Er.A[1,-4]
    return RightCompositeEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{<:Number, 4}, B::LocalOperator{1,2},Er::RightEnvironmentTensor{<:Number, 3})
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[3,-1,1,-4] * B.A[-2,2,3] * Er.A[1,2,-3]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, EnvR::RightEnvironmentTensor{<:Number, 2})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,1,-4] * EnvR.A[1,-3]
    return DenseMPOTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 5}, EnvR::RightEnvironmentTensor{<:Number, 3})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,2,1,-4] * EnvR.A[1,2,-3]
    return DenseMPOTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2,4}, A::DenseMPOTensor{<:Number, 4})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-4])
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2,4}, B::DenseMPOTensor{<:Number, 4})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvR.A[-1,2,1,3] * B'.A[1,3,2,4] * B.A[-2,4,-3,-4])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, Λ::DenseMPOTensor{<:Number, 2})
    return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-1,-2,1,-4]*Λ.A[1,-3])
end

function contract(EnvL::RightCompositeEnvironmentTensor{<:Number, 2, 4}, Λ::DenseMPOTensor{<:Number, 2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3,-4])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 4}, A::AdjointMPOTensor{<:Number, 4})
    @tensor tmp[-1;-2] ≔ EnvL.A[1,2,-2,3] * A.A[-1,3,2,1] 
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 4}, A::AdjointMPOTensor{<:Number, 4})
    @tensor tmp[-1;-2] ≔ EnvR.A[-1,2,1,3] * A.A[1,3,2,-2] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{<:Number, 2}, EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 4})
    @tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,1] * EnvR.A[1,-1,-3,-4]
    return DenseMPOTensor(tmp)
end



function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2,5}, A::DenseMPOTensor{<:Number, 4})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[1,2,-3,-4,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-5])
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2,5}, B::DenseMPOTensor{<:Number, 4})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ EnvR.A[-1,-2,2,1,3] * B'.A[1,3,2,4] * B.A[-3,4,-4,-5])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 5}, Λ::DenseMPOTensor{<:Number, 2})
    return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[-1,-2,-3,1,-5]*Λ.A[1,-4])
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 5}, Λ::DenseMPOTensor{<:Number, 2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ Λ.A[-1,1]*EnvR.A[1,-2,-3,-4,-5])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{<:Number, 2, 5}, A::AdjointMPOTensor{<:Number, 4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[1,2,-2,-3,3] * A.A[-1,3,2,1] 
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 5}, A::AdjointMPOTensor{<:Number, 4})
    @tensor tmp[-1 -2;-3] ≔ EnvR.A[-1,-2,2,1,3] * A.A[1,3,2,-3] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{<:Number, 3}, EnvR::RightCompositeEnvironmentTensor{<:Number, 2, 5})
    @tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,2,1] * EnvR.A[1,2,-1,-3,-4]
    return DenseMPOTensor(tmp)
end
