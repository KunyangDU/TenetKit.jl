
# join

function contract(EnvL::SparseLeftEnvironmentTensor{1}, EnvR::SparseRightEnvironmentTensor{1}, lm::LayerMap)
    mps = nothing
    Nthr = get_nworker()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(lm.rev) && break
                ind = [t[1] for t in lm.rev[ct]]
                isempty(ind) && continue
                C = contract(EnvL.A[ct], sum(EnvR.A[ind]))
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
        for (i, tuples) in enumerate(lm.rev)
            isempty(tuples) && continue
            ind = [t[1] for t in tuples]
            mps = axpy!(1, contract(EnvL.A[i], sum(EnvR.A[ind])), mps)
        end
    end
    return mps
end

contract(EnvL::LeftEnvironmentTensor{3}, EnvR::RightCompositeEnvironmentTensor{1, 4}) = MPSTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,2,1] * EnvR.A[1,2,-2,-3])
contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{1, 3}) = MPSTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,1] * EnvR.A[1,-2,-3])
contract(El::LeftCompositeEnvironmentTensor{2,3},Er::RightEnvironmentTensor{2}) = MPSTensor(@tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,1] * Er.A[1,-3])
contract(El::LeftCompositeEnvironmentTensor{2,4},Er::RightEnvironmentTensor{3}) = MPSTensor(@tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,2,1] * Er.A[1,2,-3])

contract(EnvL::LeftEnvironmentTensor{3}, EnvR::RightCompositeEnvironmentTensor{2, 5}) = DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,2,1] * EnvR.A[1,2,-1,-3,-4])
contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{2, 4}) = DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,1] * EnvR.A[1,-1,-3,-4])
contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightEnvironmentTensor{2}) = DenseMPOTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,1,-4] * EnvR.A[1,-3])
contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, EnvR::RightEnvironmentTensor{3}) = DenseMPOTensor(@tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,2,1,-4] * EnvR.A[1,2,-3])

contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightEnvironmentTensor{2}, ::Nothing) = contract(EnvL,EnvR)
contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, EnvR::RightEnvironmentTensor{3}, ::Nothing) = contract(EnvL,EnvR)
contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{2, 4},::Nothing) = contract(EnvL,EnvR)
contract(El::LeftEnvironmentTensor{3}, Er::RightCompositeEnvironmentTensor{2, 5}, ::Nothing) = contract(El,Er)
contract(El::LeftCompositeEnvironmentTensor{2, 4}, Er::RightEnvironmentTensor{3}, ::Nothing) = contract(El,Er)
contract(El::LeftEnvironmentTensor{3}, Er::RightCompositeEnvironmentTensor{1, 4}, ::Nothing) = contract(El,Er)
