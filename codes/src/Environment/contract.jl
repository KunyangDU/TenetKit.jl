
#= PUSH ENVIRONMENT =#
"""
MPS + ENVR
push left 
"""
function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * B.A[3,-2,2] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * mpot.A[4,2] * B.A[3,-2,4] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{2,1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,2,1] * mpot.A[4,-2,2] * B.A[3,-3,4] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ A.A[-1,2,1] * B.A[4,-3,2] * EnvR.A[1,-2,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,2},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[-1,2,1] * mpot.A[5,3,2] * B.A[4,-2,5] * EnvR.A[1,3,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[3,4,-2] * mpot.A[2,4] * B.A[-1,1,2] * EnvL.A[1,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[3,2,-2] * B.A[-1,1,2] * EnvL.A[1,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,2},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2 -3] ≔ A.A[3,4,-3] * mpot.A[2,-2,4] * B.A[-1,1,2] * EnvL.A[1,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},::IdentityOperator{1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ A.A[3,2,-3] * B.A[-1,1,2] * EnvL.A[1,-2,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{2,1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[4,5,-2] * mpot.A[3,2,5] * B.A[-1,1,3] * EnvL.A[1,2,4]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, mpot::LocalOperator{1, 1}, B::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1,-2;-3] ≔ A.A[-1,2,1] * mpot.A[4,2] * B.A[3,-3,4] * EnvR.A[1,-2,3]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::LocalOperator{1,1},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2 -3] ≔ A.A[3,4,-3] * mpot.A[2,4] * B.A[-1,1,2] * EnvL.A[1,-2,3]
    return LeftEnvironmentTensor(tmp)
end


#= PUSH ENVIRONMENT =#
"""
MPO + sparse MPO + ENVR
push left
"""
function contract(A::DenseMPOTensor{4}, B::SparseMPOTensor{N, M}, C::AdjointMPOTensor{4}, EnvR::SparseRightEnvironmentTensor) where {N,M}
    @assert EnvR.D[1] == M
    tmpEnvR = Vector{Any}(nothing,N)
    for i in 1:N, j in 1:M
        isnothing(B.m[i,j]) && continue
        if isnothing(tmpEnvR[i])
            tmpEnvR[i] = contract(A, B.m[i,j], C, EnvR.A[j])
        else 
            tmpEnvR[i] += contract(A, B.m[i,j], C, EnvR.A[j])
        end
    end
    return convert(Vector{RightEnvironmentTensor},tmpEnvR)
end

function contract(A::DenseMPOTensor{4}, B::SparseMPOTensor{N, M}, C::AdjointMPOTensor{4}, EnvL::SparseLeftEnvironmentTensor) where {N,M}
    @assert EnvL.D[1] == N
    tmpEnvL = Vector{Any}(nothing,M)
    for i in 1:N, j in 1:M
        isnothing(B.m[i,j]) && continue
        if isnothing(tmpEnvL[j])
            tmpEnvL[j] = contract(EnvL.A[i], A, B.m[i,j], C)
        else 
            tmpEnvL[j] += contract(EnvL.A[i], A, B.m[i,j], C)
        end
    end
    return convert(Vector{LeftEnvironmentTensor},tmpEnvL)
end

"""
ENVL + MPO + sparse MPO
push right
"""
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,1] * A.A[2,1,-3,5] * B.A[4,-2,2] * C.A[-1,5,4,3]
    return LeftEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[4,2,1] * A.A[3,1,-2,6] * B.A[5,2,3] * C.A[-1,6,5,4]
    return LeftEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[3,1] * A.A[2,1,-2,5] * B.A[4,2] * C.A[-1,5,4,3]
    return LeftEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, ::IdentityOperator{1}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[3,1] * A.A[2,1,-2,4] * C.A[-1,4,2,3]
    return LeftEnvironmentTensor(tmp)
end
function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, ::IdentityOperator{1}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,-2,1] * A.A[2,1,-3,4] * C.A[-1,4,2,3]
    return LeftEnvironmentTensor(tmp)
end
"""
MPOs + ENVR
push left
"""
function contract(A::DenseMPOTensor{4}, ::IdentityOperator{1}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[2,-1,1,4] * C.A[3,4,2,-2] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[2,-1,1,5] * B.A[4,2] * C.A[3,5,4,-2] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,5] * B.A[4,-2,2] * C.A[3,5,4,-3] * EnvR.A[1,3]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[3,-1,1,6] * B.A[5,2,3] * C.A[4,6,5,-2] * EnvR.A[1,2,4]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, ::IdentityOperator{1}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,4] * C.A[3,4,2,-3] * EnvR.A[1,-2,3]
    return RightEnvironmentTensor(tmp)
end
function contract(A::DenseMPOTensor{4}, B::LocalOperator{1,1}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,4] * B.A[5,2] * C.A[3,4,5,-3] * EnvR.A[1,-2,3]
    return RightEnvironmentTensor(tmp)
end



contract(A::DenseMPOTensor{4}, B::LocalOperator{1,2}, C::AdjointMPOTensor{4},EnvL::LeftEnvironmentTensor{2}) = contract(EnvL,A,B,C)
contract(A::DenseMPOTensor{4}, B::LocalOperator{2,1}, C::AdjointMPOTensor{4},EnvL::LeftEnvironmentTensor{3}) = contract(EnvL,A,B,C)
contract(A::DenseMPOTensor{4}, B::Union{LocalOperator{1,1}, IdentityOperator{1}}, C::AdjointMPOTensor{4},EnvL::LeftEnvironmentTensor{3}) = contract(EnvL,A,B,C)
contract(A::DenseMPOTensor{4}, B::Union{LocalOperator{1,1}, IdentityOperator{1}}, C::AdjointMPOTensor{4},EnvL::LeftEnvironmentTensor{2}) = contract(EnvL,A,B,C)

#= Env4 =#

function contract(EnvL::SparseLeftEnvironmentTensor{2}, Hup::SparseMPOTensor, t::DenseMPOTensor, Hdown::SparseMPOTensor, t′::AdjointMPOTensor, EnvR::SparseRightEnvironmentTensor{2})
    tmp = 0
    validinds = filter(x -> !isnothing(Hup.m[x[1],x[3]]) && !isnothing(Hdown.m[x[2],x[4]]), [(i,j,k,l) for i in 1:EnvL.D[1], j in 1:EnvL.D[2], k in 1:EnvR.D[1], l in 1:EnvR.D[2]][:])
    Nthr = get_num_threads_julia()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(validinds) && break
                i,j,k,l = validinds[ct]
                C = contract(EnvL.A[i,j],Hup.m[i,k], t ,Hdown.m[j,l], t′ ,EnvR.A[k,l])
                lock(Lock)
                try
                    tmp += C
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for (i,j,k,l) in validinds
            tmp += contract(EnvL.A[i,j],Hup.m[i,k], t ,Hdown.m[j,l], t′ ,EnvR.A[k,l])
        end
    end
    return tmp
end

function contract(EnvL::LeftEnvironmentTensor{2},ht::LocalOperator{1,1},t::DenseMPOTensor{4},hb::LocalOperator{1,1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[4,3] * ht.A[1,5] * t.A[2,3,7,1] * hb.A[6,2] * t′.A[8,5,6,4] * EnvR.A[7,8]
end

function contract(EnvL::LeftEnvironmentTensor{2},::IdentityOperator{1},t::DenseMPOTensor{4},hb::LocalOperator{1,1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[3,2] * t.A[1,2,6,4] * hb.A[5,1] * t′.A[7,4,5,3] * EnvR.A[6,7]
end

function contract(EnvL::LeftEnvironmentTensor{2},ht::LocalOperator{1,1},t::DenseMPOTensor{4},::IdentityOperator{1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[3,2] * ht.A[1,4] * t.A[5,2,6,1] * t′.A[7,4,5,3] * EnvR.A[6,7]
end

function contract(EnvL::LeftEnvironmentTensor{2},::IdentityOperator{1},t::DenseMPOTensor{4},::IdentityOperator{1},t′::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[2,1] * t.A[3,1,5,4] * t′.A[6,4,3,2] * EnvR.A[5,6]
end

function contract(A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[-1,3,1] * A′.A[2,-2,3] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::DenseLeftEnvironmentTensor, A::MPSTensor, B::AdjointMPSTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[1,2] * A.A[2,3,4] * A′.A[5,1,3] * EnvR.A[4,5]
end

