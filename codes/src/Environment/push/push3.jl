
function pushleft(A::T, mpo::SparseMPOTensor{DL,D,DR}, B::S, EnvR::SparseRightEnvironmentTensor{1}) where {DL,D,DR,T<:Union{MPSTensor{3},DenseMPOTensor{4}},S<:Union{AdjointMPSTensor{3},AdjointMPOTensor{4}}}
    @assert DR == EnvR.D[1]
    tmpEnvR = Vector{Any}(nothing, D)
    r_map = _validind1(mpo, R2L())
    validind = [(j, r_pairs) for (j, r_pairs) in enumerate(r_map) if !isempty(r_pairs)]
    threaded_foreach(validind) do (j, r_pairs)
        weighted_env = sum(w * EnvR[b] for (b, w) in r_pairs)
        tmpEnvR[j] = contract(A, mpo[j], B, weighted_env)
    end
    return SparseRightEnvironmentTensor(convert(Vector{RightEnvironmentTensor}, tmpEnvR))
end

function pushright(A::T, mpo::SparseMPOTensor{DL,D,DR}, B::S, EnvL::SparseLeftEnvironmentTensor{1}) where {DL,D,DR,T<:Union{MPSTensor{3},DenseMPOTensor{4}},S<:Union{AdjointMPSTensor{3},AdjointMPOTensor{4}}}
    @assert DL == EnvL.D[1]
    tmpEnvL = Vector{Any}(nothing, D)
    l_map = _validind1(mpo, L2R())
    validind = [(j, l_pairs) for (j, l_pairs) in enumerate(l_map) if !isempty(l_pairs)]
    threaded_foreach(validind) do (j, l_pairs)
        weighted_env = sum(w * EnvL[b] for (b, w) in l_pairs)
        tmpEnvL[j] = contract(A, mpo[j], B, weighted_env)
    end
    return SparseLeftEnvironmentTensor(convert(Vector{LeftEnvironmentTensor}, tmpEnvL))
end
