
#= Env4 =#

function pushright(Hup::SparseMPO, ρ::DenseMPO, Hdown::SparseMPO, ρ′::AdjointMPO, EnvL::SparseLeftEnvironmentTensor{2}, site::Int64)
    tmpEnvL = Array{Any}(nothing, Hup.D[site][2], Hdown.D[site][2])
    l_map_up = _validind1(Hup[site], L2R())
    l_map_down = _validind1(Hdown[site], L2R())
    vind_up = [(j, l_pairs) for (j, l_pairs) in enumerate(l_map_up) if !isempty(l_pairs)]
    vind_down = [(j, l_pairs) for (j, l_pairs) in enumerate(l_map_down) if !isempty(l_pairs)]
    pairs = [(a,b) for a in eachindex(vind_up) for b in eachindex(vind_down)]
    threaded_foreach(pairs) do (a, b)
        op_up, l_pairs_up = vind_up[a]
        op_down, l_pairs_down = vind_down[b]
        weighted_env = sum(w_i * w_j * EnvL.A[i,j] for (i, w_i) in l_pairs_up for (j, w_j) in l_pairs_down)
        tmpEnvL[op_up,op_down] = pushright(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], weighted_env)
    end
    return SparseLeftEnvironmentTensor(convert(Array{LeftEnvironmentTensor}, tmpEnvL))
end

function pushleft(Hup::SparseMPO, ρ::DenseMPO, Hdown::SparseMPO, ρ′::AdjointMPO, EnvR::SparseRightEnvironmentTensor{2}, site::Int64)
    tmpEnvR = Array{Any}(nothing, Hup.D[site][2], Hdown.D[site][2])
    r_map_up = _validind1(Hup[site], R2L())
    r_map_down = _validind1(Hdown[site], R2L())
    vind_up = [(j, r_pairs) for (j, r_pairs) in enumerate(r_map_up) if !isempty(r_pairs)]
    vind_down = [(j, r_pairs) for (j, r_pairs) in enumerate(r_map_down) if !isempty(r_pairs)]
    pairs = [(a,b) for a in eachindex(vind_up) for b in eachindex(vind_down)]
    threaded_foreach(pairs) do (a, b)
        op_up, r_pairs_up = vind_up[a]
        op_down, r_pairs_down = vind_down[b]
        weighted_env = sum(w_k * w_l * EnvR.A[k,l] for (k, w_k) in r_pairs_up for (l, w_l) in r_pairs_down)
        tmpEnvR[op_up,op_down] = pushleft(Hup[site][op_up], ρ[site], Hdown[site][op_down], ρ′[site], weighted_env)
    end
    return SparseRightEnvironmentTensor(convert(Array{RightEnvironmentTensor}, tmpEnvR))
end

pushright(::Nothing, A::DenseMPOTensor{4}, h::AbstractLocalOperator, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2}) = contract(A,h,A′,EnvL)
function pushright(h::LocalOperator{1, 1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ h.A[1,5] * A.A[4,2,-2,1] * A′.A[-1,5,4,3] * EnvL.A[3,2]
    return LeftEnvironmentTensor(tmp)
end
function pushright(::IdentityOperator{1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,1,-2,3] * A′.A[-1,3,4,2] * EnvL.A[2,1]
    return LeftEnvironmentTensor(tmp)
end


pushleft(::Nothing, A::DenseMPOTensor{4}, h::AbstractLocalOperator, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2}) = contract(A,h,A′,EnvR)
function pushleft(h::LocalOperator{1, 1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ h.A[1,4] * A.A[5,-1,2,1] * A′.A[3,4,5,-2] * EnvR.A[2,3]
    return RightEnvironmentTensor(tmp)
end
function pushleft(::IdentityOperator{1}, A::DenseMPOTensor{4}, ::Nothing, A′::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor tmp[-1;-2] ≔ A.A[4,-1,1,3] * A′.A[2,3,4,-2] * EnvR.A[1,2]
    return RightEnvironmentTensor(tmp)
end
##
function pushleft(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor x[-1;-2] ≔ ht.A[1,6] * objt.A[2,-1,3,1] * hb.A[5,2] * objb.A[4,6,5,-2] * EnvR.A[3,4]
    return RightEnvironmentTensor(x)
end

function pushleft(::IdentityOperator{1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    # @tensor x[-1;-2] ≔ objt.A[2,-1,3,1] * hb.A[5,2] * objb.A[4,1,5,-2] * EnvR.A[3,4]
    @tensor x[-1;-2] ≔ objt.A[1,-1,2,4] * hb.A[5,1] * objb.A[3,4,5,-2] * EnvR.A[2,3]
    return RightEnvironmentTensor(x)
end

function pushleft(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    # @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[2,-1,3,1] * objb.A[4,5,2,-2] * EnvR.A[3,4]
    @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[4,-1,2,1] * objb.A[3,5,4,-2] * EnvR.A[2,3]
    return RightEnvironmentTensor(x)
end

function pushleft(::IdentityOperator{1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor x[-1;-2] ≔ objt.A[3,-1,1,4] * objb.A[2,4,3,-2] * EnvR.A[1,2]
    return RightEnvironmentTensor(x)
end
