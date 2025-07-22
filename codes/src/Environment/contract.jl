
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

