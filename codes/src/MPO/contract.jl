#= PUSH ENVIRONMENT =#
"""
MPO + sparse MPO + ENVR
push left
"""
function contract(A::DenseMPOTensor{4}, B::SparseMPOTensor{N, M}, C::AdjointMPOTensor{4}, EnvR::SparseRightEnvironmentTensor) where {N,M}
    @assert EnvR.D == M
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
    @assert EnvL.D == N
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
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{3}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,1] * A.A[2,1,-3,5] * B.A[4,-2,2] * C.A[-1,5,4,3]
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{3}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[4,2,1] * A.A[3,1,-2,6] * B.A[5,2,3] * C.A[-1,6,5,4]
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{2}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[3,1] * A.A[2,1,-2,5] * B.A[4,2] * C.A[-1,5,4,3]
    return LeftEnvironmentTensor(tmp)
end

"""
MPOs + ENVR
push left
"""
function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{2}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[2,-1,1,5] * B.A[3,2] * C.A[4,5,3,-2] * EnvR.A[1,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{3}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ A.A[2,-1,1,5] * B.A[3,-2,2] * C.A[4,5,3,-3] * EnvR.A[1,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{3}, C::AdjointMPOTensor{4},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[3,-1,1,6] * B.A[5,2,3] * C.A[4,6,5,-2] * EnvR.A[1,2,4]
    return RightEnvironmentTensor(tmp)
end
"""
ENVL + MPO
push right
"""
function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{2}, C::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[3,-2,1] * A.A[2,1,-3,5] * B.A[4,2] * C.A[-1,5,4,3]
    return LeftEnvironmentTensor(tmp)
end
"""
MPO + ENVR
push left
"""
function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{2}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[3,-1,1,5] * B.A[6,3] * C.A[4,5,6,-3] * EnvR.A[1,-2,4]
    return RightEnvironmentTensor(tmp)
end

#= COMPOSITE ENVIRONMENT =#
"""
MPO + ENVR
make composite ENVR (MPO + ENVR 1)
"""
function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,-1,1,3] * B.A[2,3,4,-2] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[3,-1,1,5] * B.A[6,-2,2,3] * C.A[4,5,6,-3] * EnvR.A[1,2,4]
    return RightEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,2,-2,3] * B.A[-1,3,4,1] * EnvL.A[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1 ;-2 -3] ≔ A.A[3,4,-3,5] * B.A[6,2,-2,3] * C.A[-1,5,6,1] * EnvL.A[1,2,4]
    return LeftEnvironmentTensor(tmp)
end

"""
ENVL + MPO
make composite ENVL (ENVL + MPO 1)
"""
function contract(El::LeftEnvironmentTensor{2},A::DenseMPOTensor{4}, B::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[2,1,-3,-4] * B.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(El::LeftEnvironmentTensor{2},A::DenseMPOTensor{4}, B::DenseMPOTensor{3})
    @tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,1] * A.A[2,1,-4,-5] * B.A[-2,-3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
"""
MPO + ENVR
make composite ENVR (MPO + ENVR 1)
"""
function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{2},Er::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[2,-1,1,-4] * B.A[-2,2] * Er.A[1,-3]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{3},Er::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,-2,2] * Er.A[1,-4]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{3},Er::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3 -4] ≔ A.A[3,-1,1,-4] * B.A[-2,2,3] * Er.A[1,2,-3]
    return RightCompositeEnvironmentTensor(tmp)
end
"""
ENVL + MPO
make composite ENVL (ENVL + MPO 1)
"""
function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{3})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-1,2,1] * A.A[3,1,-3,-4] * B.A[-2,2,3]
    return LeftCompositeEnvironmentTensor(tmp)
end
"""
ENVL + composite MPO
make composite ENVL (ENVL + composite MPO 1)
"""
function contract(EnvL::LeftEnvironmentTensor{2}, A::CompositeMPOTensor{2, 6}, B::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,1] * A.A[-3,2,1,-5,-6,-7] * B.A[-2,-4,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, A::CompositeMPOTensor{2, 6}, B::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,2,1] * A.A[-3,3,1,-4,-5,-6] * B.A[-2,2,3]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::CompositeMPOTensor{2, 6}, B::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,1] * A.A[-3,2,1,-4,-5,-6] * B.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

"""
MPO + ENVR
make composite ENVR (MPO + ENVR)
"""
function contract(A::DenseMPOTensor{4}, B::DenseMPOTensor{2},Er::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ A.A[2,-1,1,-5] * B.A[-3,2] * Er.A[1,-2,-4]
    return RightCompositeEnvironmentTensor(tmp)
end
"""
ENVL + MPO
make composite ENVL (ENVL + MPO 1)
"""
function contract(El::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3 -4 -5] ≔ El.A[-1,-3,1] * A.A[2,1,-4,-5] * B.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

"""
ENVL + composite MPO
make composite ENVL (ENVL + composite MPO 1)
"""
function contract(EnvL::LeftEnvironmentTensor{3}, A::CompositeMPOTensor{2, 6}, B::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-4,1] * A.A[-3,2,1,-5,-6,-7] * B.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
"""
composite ENVL (ENVL + MPO 1) + MPO
make composite ENVL (ENVL + MPO 1)
"""
function contract(EnvL::LeftCompositeEnvironmentTensor{3, 7}, A::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,-2,2,1,-4,-5,-6] * A.A[-3,1,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{3, 6}, A::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-2,1,-5,-6,-7] * A.A[-3,-4,1]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{3, 6}, A::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-1,-2,1,-4,-5,-6] * A.A[-3,1]
    return LeftCompositeEnvironmentTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{3, 7}, A::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6 -7] ≔ EnvL.A[-1,-2,1,-4,-5,-6,-7] * A.A[-3,1]
    return LeftCompositeEnvironmentTensor(tmp)
end

#= EFFECTIVE MPO =#

"""
composite ENVL (ENVL + MPO 1) + composite ENVR (MPO + ENVR 1)
make eff composite MPO
"""
function contract(El::LeftCompositeEnvironmentTensor{2, 4}, Er::RightCompositeEnvironmentTensor{2, 4})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ El.A[-3,-2,1,-6] * Er.A[1,-1,-4,-5]
    return CompositeMPOTensor(tmp)
end

function contract(El::LeftCompositeEnvironmentTensor{2, 5}, Er::RightCompositeEnvironmentTensor{2, 5})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ El.A[-3,-2,2,1,-6] * Er.A[1,2,-1,-4,-5]
    return CompositeMPOTensor(tmp)
end

"""
ENVL + MPO + ENVR
make eff MPO
"""
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,1] * A.A[-1,1,2,-4] * EnvR.A[2,-3]
    return DenseMPOTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,1] * A.A[-2,1,2,-6] * B.A[-1,2,3,-5] * EnvR.A[3,-4]
    return CompositeMPOTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::DenseMPOTensor{4}, D::DenseMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp1[-1 -2;-3 -4 -5] ≔ EnvL.A[-1,1,2] * A.A[-2,1,-3,3] * C.A[3,2,-4,-5]
    @tensor tmp2[-1 -2 -3;-4 -5] ≔ B.A[3,-1,1,-5] * D.A[-3,-2,2,3] * EnvR.A[1,2,-4]
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ tmp1[-3,-2,2,1,-6] * tmp2[1,2,-1,-4,-5]
    return CompositeMPOTensor(tmp)
end

"""
sparse ENVL + 2*MPO + 2*sparse MPO + sparse ENVR
make eff composite MPO
"""
function contract(EnvL::SparseLeftEnvironmentTensor, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::SparseMPOTensor{N₁,M₁}, D::SparseMPOTensor{N₂,M₂}, EnvR::SparseRightEnvironmentTensor) where {N₁,M₁,N₂,M₂}
    @assert M₁ == N₂
    tmp = nothing
    for i in 1:N₁, j in 1:M₁, k in 1:M₂
        isnothing(C.m[i,j]) | isnothing(D.m[j,k]) && continue
        tmp1 = contract(EnvL.A[i], A, C.m[i,j])
        tmp2 = contract(B, D.m[j,k], EnvR.A[k])
        if isnothing(tmp)
            tmp = contract(tmp1, tmp2)
        else
            tmp += contract(tmp1, tmp2)
        end
    end
    return tmp
end

"""
composite ENVL (ENVL + MPO 1) + ENVR
make eff MPO
"""
function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,1,-4] * EnvR.A[1,-3]
    return DenseMPOTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3 -4] ≔ EnvL.A[-2,-1,2,1,-4] * EnvR.A[1,2,-3]
    return DenseMPOTensor(tmp)
end

"""
composite ENVL (ENVL + composite MPO 1) + ENVR
make eff composite MPO
"""
function contract(EnvL::LeftCompositeEnvironmentTensor{3, 6}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,-2,-1,1,-5,-6] * EnvR.A[1,-4]
    return CompositeMPOTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{3, 7}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ EnvL.A[-3,-2,-1,2,1,-5,-6] * EnvR.A[1,2,-4]
    return CompositeMPOTensor(tmp)
end
"""
ENVL + 2*element + 2*MPO + ENVR
make eff composite element
"""
function contract(EnvL::LeftEnvironmentTensor, A::Union{MPSTensor,DenseMPOTensor}, B::Union{MPSTensor,DenseMPOTensor}, C::Union{MPSTensor,DenseMPOTensor}, D::Union{MPSTensor,DenseMPOTensor}, EnvR::RightEnvironmentTensor)
    return contract(contract(EnvL, A, C), contract(B, D, EnvR))
end

function contract(El::LeftCompositeEnvironmentTensor{3, 5}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,-2,1,-4,-5] * mpo.A[-3,1]
    return LeftCompositeEnvironmentTensor(tmp)
end

"""
sparse ENVL + MPOs + sparse ENVR
make eff MPOs
"""
function contract(EnvL::SparseLeftEnvironmentTensor, A::DenseMPOTensor{4}, B::SparseMPOTensor{N,M}, C::Union{DenseMPOTensor{4},AdjointMPOTensor{4}}, EnvR::SparseRightEnvironmentTensor) where {N,M}
    tmp = nothing
    for i in 1:N, j in 1:M
        isnothing(B.m[i,j]) && continue
        tmp1 = contract(EnvL.A[i], A, B.m[i,j], C, EnvR.A[j])
        if isnothing(tmp)
            tmp = tmp1
        else
            tmp += tmp1
        end
    end
    return tmp
end
"""
ENVL + MPOs + ENVR
make eff MPOs
"""
function contract(EnvL::DenseLeftEnvironmentTensor{3}, A::DenseMPOTensor{4}, B::DenseMPOTensor{4}, C::DenseMPOTensor{4}, D::DenseMPOTensor{4}, EnvR::DenseRightEnvironmentTensor{3})
    @tensor tmp1[-1 -2;-3 -4 -5] ≔ EnvL.A.A[-1,1,2] * A.A[-2,1,-3,3] * C.A[3,2,-4,-5]
    @tensor tmp2[-1 -2 -3;-4 -5] ≔ B.A[3,-1,1,-5] * D.A[-3,-2,2,3] * EnvR.A.A[1,2,-4]
    @tensor tmp[-1 -2 -3;-4 -5 -6] ≔ tmp1[-3,-2,2,1,-6] * tmp2[1,2,-1,-4,-5]
    return CompositeMPOTensor(tmp)
end
#= SCALAR =#

"""
ENVL + MPO + ENVR
make scalar
"""
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[2,1] * A.A[3,1,5,4] * B.A[6,4,3,2] * EnvR.A[5,6]
end

function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{2}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    return @tensor EnvL.A[3,1] * A.A[2,1,6,5] * B.A[4,2] * C.A[7,5,4,3] * EnvR.A[6,7]
end
"""
ENVL + MPO + ENVR
make scalar
"""
function contract(EnvL::LeftEnvironmentTensor{2}, A::DenseMPOTensor{4}, B::DenseMPOTensor{3}, C::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{3})
    return @tensor EnvL.A[3,1] * A.A[2,1,6,5] * B.A[4,7,2] * C.A[8,5,4,3] * EnvR.A[6,7,8]
end

#= INTRODUCTION =#

"""
WHAT IS THIS?
"""
function contract(EnvL::DenseLeftEnvironmentTensor, A::DenseMPOTensor, B::DenseMPOTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end
function contract(EnvL::DenseLeftEnvironmentTensor, A::DenseMPOTensor, B::AdjointMPOTensor, EnvR::DenseRightEnvironmentTensor)
    return contract(EnvL.A, A, B, EnvR.A)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2,4}, A::DenseMPOTensor{4})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[1,2,-3,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-4])
end

function contract(EnvR::RightCompositeEnvironmentTensor{2,4}, B::DenseMPOTensor{4})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvR.A[-1,2,1,3] * B'.A[1,3,2,4] * B.A[-2,4,-3,-4])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, Λ::DenseMPOTensor{2})
    return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-1,-2,1,-4]*Λ.A[1,-3])
end

function contract(EnvL::RightCompositeEnvironmentTensor{2, 4}, Λ::DenseMPOTensor{2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3,-4])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, A::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvL.A[1,2,-2,3] * A.A[-1,3,2,1] 
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{2, 4}, A::AdjointMPOTensor{4})
    @tensor tmp[-1;-2] ≔ EnvR.A[-1,2,1,3] * A.A[1,3,2,-2] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{2, 4})
    @tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,1] * EnvR.A[1,-1,-3,-4]
    return DenseMPOTensor(tmp)
end

function contract(tr::DenseMPOTensor{2},obj::DenseMPOTensor{4})
    return DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ tr.A[-2,1] * obj.A[-1,1,-3,-4])
end

function contract(obj::DenseMPOTensor{4},tl::DenseMPOTensor{2})
    return DenseMPOTensor(@tensor tmp[-1,-2;-3,-4] ≔ obj.A[-1,-2,1,-4] * tl.A[1,-3])
end

function splice!(obj::DenseMPO{L}, A::DenseMPOTensor{4}, csite::Int64) where L
    site = obj.center[1]
    if csite == site + 1
        @tensor tmp[-1,-2;-3,-4] ≔ obj.ts[site].A[-1,-2,4,-4] * A.A[2,4,1,3] * obj.ts[csite]'.A[1,3,2,-3]
        obj.ts[site] = DenseMPOTensor(tmp)
    elseif csite == site - 1
        @tensor tmp[-1,-2;-3,-4] ≔ obj.ts[site].A[-1,4,-3,-4] * A.A[2,1,4,3] * obj.ts[csite]'.A[-2,3,2,1]
        obj.ts[site] = DenseMPOTensor(tmp)
    else
        @error "index out of range"
    end
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2,5}, A::DenseMPOTensor{4})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[1,2,-3,-4,3] * A'.A[4,3,2,1] * A.A[-2,-1,4,-5])
end

function contract(EnvR::RightCompositeEnvironmentTensor{2,5}, B::DenseMPOTensor{4})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ EnvR.A[-1,-2,2,1,3] * B'.A[1,3,2,4] * B.A[-3,4,-4,-5])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, Λ::DenseMPOTensor{2})
    return LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3,-4,-5] ≔ EnvL.A[-1,-2,-3,1,-5]*Λ.A[1,-4])
end

function contract(EnvR::RightCompositeEnvironmentTensor{2, 5}, Λ::DenseMPOTensor{2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2,-3;-4,-5] ≔ Λ.A[-1,1]*EnvR.A[1,-2,-3,-4,-5])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 5}, A::AdjointMPOTensor{4})
    @tensor tmp[-1;-2 -3] ≔ EnvL.A[1,2,-2,-3,3] * A.A[-1,3,2,1] 
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{2, 5}, A::AdjointMPOTensor{4})
    @tensor tmp[-1 -2;-3] ≔ EnvR.A[-1,-2,2,1,3] * A.A[1,3,2,-3] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{3}, EnvR::RightCompositeEnvironmentTensor{2, 5})
    @tensor tmp[-1,-2;-3,-4] ≔ EnvL.A[-2,2,1] * EnvR.A[1,2,-1,-3,-4]
    return DenseMPOTensor(tmp)
end

#= ================================================ =#

