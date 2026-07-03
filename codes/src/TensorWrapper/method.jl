function _validind1(obj::SparseMPOTensor{DL,D,DR,T}, ::L2R) where {DL,D,DR,T}
    result = [Tuple{Int64,T}[] for _ in 1:D]
    for a in 1:DL
        for (c, (b,)) in enumerate(obj.left.fwd[a])
            push!(result[b], (a, obj.left.fwd_w[a][c]))
        end
    end
    return result
end

function _validind1(obj::SparseMPOTensor{DL,D,DR,T}, ::R2L) where {DL,D,DR,T}
    result = [Tuple{Int64,T}[] for _ in 1:D]
    for a in 1:D
        for (c, (b,)) in enumerate(obj.right.fwd[a])
            push!(result[a], (b, obj.right.fwd_w[a][c]))
        end
    end
    return result
end

# 返回 Vector{Vector{Tuple{Int64,T}}} — 算符索引 → [(bond_index, weight), ...]
function _validind2(obj::SparseMPOTensor{DL,D,DR,T}, ::R2L) where {DL,D,DR,T}
    result = [Tuple{Int64,T}[] for _ in eachindex(obj)]
    for a in 1:DL
        for (c, (b,)) in enumerate(obj.left.fwd[a])
            push!(result[b], (a, obj.left.fwd_w[a][c]))
        end
    end
    return result
end

function _validind2(obj::SparseMPOTensor{DL,D,DR,T}, ::L2R) where {DL,D,DR,T}
    result = [Tuple{Int64,T}[] for _ in eachindex(obj)]
    for a in 1:D
        for (c, (b,)) in enumerate(obj.right.fwd[a])
            push!(result[a], (b, obj.right.fwd_w[a][c]))
        end
    end
    return result
end

# 返回 (l_inds, j, r_inds, wl, wr): wl/r 与 l_inds/r_inds 并行，调用端用 sum(Env .* w)
function _validind(obj::SparseMPOTensor{DL,D,DR,T}) where {DL,D,DR,T}
    left_map = _validind2(obj, R2L())
    right_map = _validind2(obj, L2R())
    result = Tuple{Vector{Int64}, Int64, Vector{Int64}, Vector{T}, Vector{T}}[]
    for j in eachindex(obj)
        l_info = left_map[j]
        r_info = right_map[j]
        isempty(l_info) && continue
        isempty(r_info) && continue
        l_inds = [p[1] for p in l_info]
        wl = [p[2] for p in l_info]
        r_inds = [p[1] for p in r_info]
        wr = [p[2] for p in r_info]
        push!(result, (l_inds, j, r_inds, wl, wr))
    end
    return result
end

# 返回 (l_inds, (j, m), r_inds, wl, w_mid, wr): 每个 (j,m) pair 一个 entry
function _validind(obj₁::SparseMPOTensor{DL₁,D₁,DR₁,T₁}, obj₂::SparseMPOTensor{DL₂,D₂,DR₂,T₂}) where {DL₁,D₁,DR₁,DL₂,D₂,DR₂,T₁,T₂}
    @assert obj₁.right == obj₂.left "LayerMap not compatible!"
    left_map = _validind2(obj₁, R2L())
    right_map = _validind2(obj₂, L2R())
    result = Tuple{Vector{Int64}, Tuple{Int64,Int64}, Vector{Int64}, Vector{Number}, Number, Vector{Number}}[]
    for j in 1:D₁
        for (pi_mid, (m,)) in enumerate(obj₁.right.fwd[j])
            w_mid = obj₁.right.fwd_w[j][pi_mid]
            l_info = left_map[j]
            r_info = right_map[m]
            isempty(l_info) && continue
            isempty(r_info) && continue
            l_inds = [p[1] for p in l_info]
            wl = [p[2] for p in l_info]
            r_inds = [p[1] for p in r_info]
            wr = [p[2] for p in r_info]
            push!(result, (l_inds, (j, m), r_inds, wl, w_mid, wr))
        end
    end
    return result
end

# For {0} SparseProjectiveHamiltonian
function _validind0(lm::LayerMap{N,D₁,D₂,T}) where {N,D₁,D₂,T}
    result = Tuple{Vector{Int64}, Nothing, Vector{Int64}, Vector{T}, Vector{T}}[]
    for i in 1:D₁
        isempty(lm.fwd[i]) && continue
        r_inds = Int64[t[end] for t in lm.fwd[i]]
        wr = lm.fwd_w[i]
        push!(result, ([i], nothing, r_inds, [one(T)], wr))
    end
    return result
end

function _wsum(Env, inds::Vector{Int64}, w::Vector{T}) where T
    # result = w[1] * Env[inds[1]]
    # for i in 2:length(inds)
    #     result = axpy!(w[i], Env[inds[i]], result)
    # end
    return sum(w .* Env[inds])
end
