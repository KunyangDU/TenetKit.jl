
#= PUSH ENVIRONMENT =#
"""
MPS + ENVR
push left 
"""
function contract(A::MPSTensor{3},mpot::DenseMPOTensor{2},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[-1,4,1] * mpot.A[3,4] * B.A[2,-2,3] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{3},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,4,1] * mpot.A[3,-2,4] * B.A[2,-3,3] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{2},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ A.A[-1,4,1] * mpot.A[3,4] * B.A[2,-3,3] * EnvR.A[1,-2,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{3},B::AdjointMPSTensor{3},EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[-1,4,1] * mpot.A[3,5,4] * B.A[2,-2,3] * EnvR.A[1,5,2]
    return RightEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{2},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[2,4,-2] * mpot.A[3,4] * B.A[-1,1,3] * EnvL.A[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{3},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2 -3] ≔ A.A[2,4,-3] * mpot.A[3,-2,4] * B.A[-1,1,3] * EnvL.A[1,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{2},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1 -2 ; -3] ≔ A.A[2,4,-3] * mpot.A[3,4] * B.A[-1,1,3] * EnvL.A[1,-2,2]
    return LeftEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3},mpot::DenseMPOTensor{3},B::AdjointMPSTensor{3},EnvL::LeftEnvironmentTensor{3})
    @tensor tmp[-1;-2] ≔ A.A[2,4,-2] * mpot.A[3,5,4] * B.A[-1,1,3] * EnvL.A[1,5,2]
    return LeftEnvironmentTensor(tmp)
end

#= COMPOSITE ENVIRONMENT =#
"""
ENVL + MPS
make composite ENVL (ENVL + MPS 1) 
"""
function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,1] * A.A[1,2,-3] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{2},A::MPSTensor{3}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,1] * A.A[1,2,-4] * mpo.A[-2,-3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2;-3 -4] ≔ El.A[-1,-3,1] * A.A[1,2,-4] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::MPSTensor{3}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,3,1] * A.A[1,2,-3] * mpo.A[-2,3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
"""
ENVL + composite MPS 
make composite ENVL (ENVL + composite MPS) 
"""
function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,1] * A.A[1,2,-3,-4] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{2},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,1] * A.A[1,2,-3,-5] * mpo.A[-2,-4,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,3,1] * A.A[1,2,-3,-4] * mpo.A[-2,3,2]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftEnvironmentTensor{3},A::CompositeMPSTensor{2, 4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,-4,1] * A.A[1,2,-3,-5] * mpo.A[-2,2]
    return LeftCompositeEnvironmentTensor(tmp)
end

"""
composite ENV (ENVL + MPS 1) +
make composite ENVL (ENVL + MPS 2)
"""
function contract(El::LeftCompositeEnvironmentTensor{3,5}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,-2,1,2,-4] * mpo.A[-3,2,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,4}, mpo::DenseMPOTensor{3})
    @tensor tmp[-1 -2 -3;-4 -5] ≔ El.A[-1,-2,1,-5] * mpo.A[-3,-4,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{3,4}, mpo::DenseMPOTensor{2})
    @tensor tmp[-1 -2 -3;-4] ≔ El.A[-1,-2,1,-4] * mpo.A[-3,1]
    return LeftCompositeEnvironmentTensor(tmp)
end
"""
composite ENV (ENVL + MPS 2) + ENVR
make composite MPS
"""
function contract(El::LeftCompositeEnvironmentTensor{3,4}, Er::RightEnvironmentTensor{2})
    return CompositeMPSTensor(El.A*Er.A)
end
function contract(El::LeftCompositeEnvironmentTensor{3,5}, Er::RightEnvironmentTensor{3})
    return CompositeMPSTensor(El.A*permute(Er.A,(2,1),(3,)))
end

"""
MPS + ENVR
make composite ENVR (MPS + ENVR 1)
"""
function contract(A::MPSTensor{3}, B::DenseMPOTensor{2}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,2,1] * B.A[-2,2] * EnvR.A[1,-3]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, B::DenseMPOTensor{3}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ A.A[-1,3,1] * B.A[-2,2,3] * EnvR.A[1,2,-3]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, B::DenseMPOTensor{2}, EnvR::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,2,1] * B.A[-3,2] * EnvR.A[1,-2,-4]
    return RightCompositeEnvironmentTensor(tmp)
end

function contract(A::MPSTensor{3}, B::DenseMPOTensor{3}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2 -3;-4] ≔ A.A[-1,2,1] * B.A[-3,-2,2] * EnvR.A[1,-4]
    return RightCompositeEnvironmentTensor(tmp)
end

#= EFFECTIVE MPS =#

"""
composite ENVL (ENVL + MPS 1) + composite ENVR (MPS + ENVR 1)
make eff composite MPS
"""
function contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, EnvR::RightCompositeEnvironmentTensor{1, 3})
    @tensor tmp[-1 -2 -3;-4] ≔ EnvL.A[-1,-2,1] * EnvR.A[1,-3,-4]
    return CompositeMPSTensor(tmp)
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 4}, EnvR::RightCompositeEnvironmentTensor{1, 4})
    @tensor tmp[-1 -2 -3;-4] ≔ EnvL.A[-1,-2,2,1] * EnvR.A[1,2,-3,-4]
    return CompositeMPSTensor(tmp)
end

"""
composite ENVL (ENVL + MPS 1)+ ENVR
make eff MPS
"""
function contract(El::LeftCompositeEnvironmentTensor{2,3},Er::RightEnvironmentTensor{2})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,1] * Er.A[1,-3]
    return MPSTensor(tmp)
end
function contract(El::LeftCompositeEnvironmentTensor{2,4},Er::RightEnvironmentTensor{3})
    @tensor tmp[-1 -2;-3] ≔ El.A[-1,-2,2,1] * Er.A[1,2,-3]
    return MPSTensor(tmp)
end
"""
r3 MPS + r2 MPS
connect a rank 3 MPS with a rank 2 MPS
"""
function contract(tr::MPSTensor{2},obj::MPSTensor{3})
    return MPSTensor(@tensor tmp[-1,-2;-3] ≔ tr.A[-1,1] * obj.A[1,-2,-3])
end

function contract(obj::MPSTensor{3},tl::MPSTensor{2})
    return MPSTensor(@tensor tmp[-1,-2;-3] ≔ obj.A[-1,-2,1] * tl.A[1,-3])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, A::AdjointMPSTensor{3})
    @tensor tmp[-1;-2] ≔ EnvL.A[1,2,-2] * A.A[-1,1,2] 
    return LeftEnvironmentTensor(tmp)
end

function contract(EnvR::RightCompositeEnvironmentTensor{1, 3}, A::AdjointMPSTensor{3})
    @tensor tmp[-1;-2] ≔ EnvR.A[-1,2,1] * A.A[1,-2,2] 
    return RightEnvironmentTensor(tmp)
end

function contract(EnvL::LeftEnvironmentTensor{2}, EnvR::RightCompositeEnvironmentTensor{1, 3})
    @tensor tmp[-1,-2;-3] ≔ EnvL.A[-1,1] * EnvR.A[1,-2,-3]
    return MPSTensor(tmp)
end

function contract(EnvL::SparseLeftEnvironmentTensor,EnvR::SparseRightEnvironmentTensor)
    @assert (w = EnvL.D) == EnvR.D
    mps = nothing 

    for i in 1:w 
        tmp = contract(EnvL.A[i],EnvR.A[i])
        if isnothing(mps)
            mps = tmp 
        else
            mps += tmp
        end
    end

    return mps
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2,3}, A::MPSTensor{3})
    LeftCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvL.A[1,2,-3] * A'.A[3,1,2] * A.A[-1,-2,3])
end

function contract(EnvR::RightCompositeEnvironmentTensor{1,3}, B::MPSTensor{3})
    RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ EnvR.A[-1,2,1] * B'.A[1,3,2] * B.A[3,-2,-3])
end

function contract(EnvL::LeftCompositeEnvironmentTensor{2, 3}, Λ::MPSTensor{2})
    return LeftCompositeEnvironmentTensor(EnvL.A*Λ.A)
end

function contract(EnvL::RightCompositeEnvironmentTensor{1, 3}, Λ::MPSTensor{2})
    return RightCompositeEnvironmentTensor(@tensor tmp[-1,-2;-3] ≔ Λ.A[-1,1]*EnvL.A[1,-2,-3])
end

function splice(Envorth::SparseLeftEnvironmentTensor,Λ::Union{DenseMPOTensor{2},MPSTensor{2}})
    tmp = Vector{LeftCompositeEnvironmentTensor}(undef,Envorth.D)
    for i in 1:Envorth.D
        tmp[i] = contract(Envorth.A[i],Λ)
    end
    return SparseLeftEnvironmentTensor(tmp)
end

function splice(Envorth::SparseRightEnvironmentTensor,Λ::Union{DenseMPOTensor{2},MPSTensor{2}})
    tmp = Vector{RightCompositeEnvironmentTensor}(undef,Envorth.D)
    for i in 1:Envorth.D
        tmp[i] = contract(Envorth.A[i],Λ)
    end
    return SparseRightEnvironmentTensor(tmp)
end

function splice(Envorth::SparseLeftEnvironmentTensor,Λ::Union{AdjointMPOTensor{4},AdjointMPSTensor{3}})
    tmp = Vector{LeftEnvironmentTensor}(undef,Envorth.D)
    for i in 1:Envorth.D
        tmp[i] = contract(Envorth.A[i],Λ)
    end
    return SparseLeftEnvironmentTensor(tmp)
end

function splice(Envorth::SparseRightEnvironmentTensor,Λ::Union{AdjointMPOTensor{4},AdjointMPSTensor{3}})
    tmp = Vector{RightEnvironmentTensor}(undef,Envorth.D)
    for i in 1:Envorth.D
        tmp[i] = contract(Envorth.A[i],Λ)
    end
    return SparseRightEnvironmentTensor(tmp)
end

function splice!(Envorth::Union{SparseLeftEnvironmentTensor,SparseRightEnvironmentTensor},
    Λ::Union{MPSTensor{2},AdjointMPSTensor{3},DenseMPOTensor{2},AdjointMPOTensor{4}})
    Envorth.A = splice(Envorth,Λ).A
end

function splice!(obj::DenseMPS{L}, A::MPSTensor{3}, csite::Int64) where L
    site = obj.center[1]
    if csite == site + 1
        @tensor tmp[-1,-2;-3] ≔ obj.ts[site].A[-1,-2,1] * A.A[1,3,2] * obj.ts[csite]'.A[2,-3,3]
        obj.ts[site] = MPSTensor(tmp)
    elseif csite == site - 1
        @tensor tmp[-1,-2;-3] ≔ obj.ts[site].A[1,-2,-3] * A.A[2,3,1] * obj.ts[csite]'.A[-1,2,3]
        obj.ts[site] = MPSTensor(tmp)
    else
        @error "index out of range"
    end
end

#= ================================================ =#

