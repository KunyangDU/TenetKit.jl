function _validind1(obj::SparseMPOTensor, ::L2R)
    result = [Int64[] for _ in eachindex(obj)]
    for (a, b) in obj.left[:]
        push!(result[b], a)
    end
    return result
end

function _validind1(obj::SparseMPOTensor, ::R2L)
    result = [Int64[] for _ in eachindex(obj)]
    for (a, b) in obj.right[:]
        push!(result[a], b)
    end
    return result
end
# 返回Vector{(ind_l, i, ind_r)}，contract(...,sum(EnvL[ind_l]),h[i],sum(EnvR[ind_r]))
function _validind(obj::SparseMPOTensor)
    left_map = _validind2(obj, R2L())
    right_map = _validind2(obj, L2R())
    return filter(x -> !isempty(x[1]) && !isempty(x[3]), [(left_map[j], j, right_map[j]) for j in eachindex(obj)])
end

# 返回Vector{(ind_l, (i,j), ind_r)}，contract(...,sum(EnvL[ind_l]),hl[i],hr[j],sum(EnvR[ind_r]))
function _validind(obj₁::SparseMPOTensor, obj₂::SparseMPOTensor)
    @assert obj₁.right == obj₂.left "LayerMap not compatible!"
    left_map = _validind2(obj₁, R2L())
    right_map = _validind2(obj₂, L2R())
    result = Tuple{Vector{Int64}, Tuple{Int64,Int64}, Vector{Int64}}[]
    for (j, m) in obj₁.right[:]
        l_inds, r_inds = left_map[j], right_map[m]
        isempty(l_inds) || isempty(r_inds) || push!(result, (l_inds, (j, m), r_inds))
    end
    return result
end

function _validind2(obj::SparseMPOTensor, ::R2L)
    result = [Int64[] for _ in eachindex(obj)]
    for (a, b) in obj.left[:]
        push!(result[b], a)
    end
    return result
end

function _validind2(obj::SparseMPOTensor, ::L2R)
    result = [Int64[] for _ in eachindex(obj)]
    for (a, b) in obj.right[:]
        push!(result[a], b)
    end
    return result
end

# For {0} SparseProjectiveHamiltonian: 通过 LayerMap 配对 EnvL 与 EnvR 的算符索引
function _validind0(lm::LayerMap{N,D₁,D₂}) where {N,D₁,D₂}
    result = Tuple{Vector{Int64}, Nothing, Vector{Int64}}[]
    for i in 1:D₁
        isempty(lm.fwd[i]) && continue
        r_inds = Int64[t[end] for t in lm.fwd[i]]
        push!(result, ([i], nothing, r_inds))
    end
    return result
end