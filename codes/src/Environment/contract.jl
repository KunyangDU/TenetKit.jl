
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
    @tensor tmp[-1;-2 -3] ≔ A.A[3,2,-3] * B.A[-1,1,2] * EnvL.A[1,-2,3]
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
function contract(A::DenseMPOTensor{4}, B::SparseMPOTensor{DL, D, DR, T}, C::AdjointMPOTensor{4}, EnvR::SparseRightEnvironmentTensor) where {DL, D, DR, T}
    @assert EnvR.D[1] == DR
    tmpEnvR = Vector{Any}(nothing, D)
    r_map = _validind1(B, R2L())
    for (j, r_pairs) in enumerate(r_map)
        isempty(r_pairs) && continue
        weighted_env = sum(w * EnvR[b] for (b, w) in r_pairs)
        tmpEnvR[j] = axpy!(1, contract(A, B[j], C, weighted_env), tmpEnvR[j])
    end
    return convert(Vector{RightEnvironmentTensor}, tmpEnvR)
end

function contract(A::DenseMPOTensor{4}, B::SparseMPOTensor{DL, D, DR, T}, C::AdjointMPOTensor{4}, EnvL::SparseLeftEnvironmentTensor) where {DL, D, DR, T}
    @assert EnvL.D[1] == DL
    tmpEnvL = Vector{Any}(nothing, D)
    l_map = _validind1(B, L2R())
    for (j, l_pairs) in enumerate(l_map)
        isempty(l_pairs) && continue
        weighted_env = sum(w * EnvL[b] for (b, w) in l_pairs)
        tmpEnvL[j] = axpy!(1, contract(weighted_env, A, B[j], C), tmpEnvL[j])
    end
    return convert(Vector{LeftEnvironmentTensor}, tmpEnvL)
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
    vind_up = _validind(Hup)
    vind_down = _validind(Hdown)
    Nthr = get_nworker()
    if Nthr > 1
        Lock = Threads.ReentrantLock()
        counter = Threads.Atomic{Int64}(1)
        pairs = [(a,b) for a in eachindex(vind_up) for b in eachindex(vind_down)]
        Threads.@sync for _ in 1:Nthr
            Threads.@spawn while true
                ct = Threads.atomic_add!(counter, 1)
                ct > length(pairs) && break
                a, b = pairs[ct]
                l_up, op_up, r_up, wl_up, wr_up = vind_up[a]
                l_down, op_down, r_down, wl_down, wr_down = vind_down[b]
                lock(Lock)
                try
                    for (pi, i) in enumerate(l_up), (pj, j) in enumerate(l_down)
                        for (pk, k) in enumerate(r_up), (pl, l) in enumerate(r_down)
                            w = wl_up[pi] * wl_down[pj] * wr_up[pk] * wr_down[pl]
                            tmp += w * contract(EnvL.A[i,j], Hup[op_up], t, Hdown[op_down], t′, EnvR.A[k,l])
                        end
                    end
                catch
                    rethrow()
                finally
                    unlock(Lock)
                end
            end
        end
    else
        for (l_up, op_up, r_up, wl_up, wr_up) in vind_up, (l_down, op_down, r_down, wl_down, wr_down) in vind_down
            for (pi, i) in enumerate(l_up), (pj, j) in enumerate(l_down)
                for (pk, k) in enumerate(r_up), (pl, l) in enumerate(r_down)
                    w = wl_up[pi] * wl_down[pj] * wr_up[pk] * wr_down[pl]
                    tmp += w * contract(EnvL.A[i,j], Hup[op_up], t, Hdown[op_down], t′, EnvR.A[k,l])
                end
            end
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

function contract(A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[2,3,-2] * A′.A[-1,1,3] * EnvL.A[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvL::DenseLeftEnvironmentTensor, A::MPSTensor, B::AdjointMPSTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::MPSTensor{3}, A′::AdjointMPSTensor{3}, EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[1,2] * A.A[2,3,4] * A′.A[5,1,3] * EnvR.A[4,5]
end



function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[3,-1,1,5] * B.A[2,6,3,-2] * C.A[4,5,6,-3] * EnvR.A[1,2,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, C::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1 ;-2 -3] ≔ A.A[6,4,-3,5] * B.A[-2,3,6,2] * C.A[-1,5,3,1] * EnvL.A[1,2,4]
    return LeftEnvironmentTensor(tmp)
end

