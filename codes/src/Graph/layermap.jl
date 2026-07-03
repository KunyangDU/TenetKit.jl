# LayerMap{N,D₁,D₂,T}: N 阶层间映射，编码 A → B 经过 N 步的路径关系
#   D₁ = 入射维数 (|A|), D₂ = 出射维数 (|B|)
#   N=1: 相邻层直连映射，路径元组长度 1
#   N≥2: compose 产生的多阶层间映射，路径保留全部中间节点
#   T  = weight 类型 (Number)
#
#   fwd[a]: 从源节点 a 出发的全部路径，每条路径 = N 元组 (n₁, n₂, ..., n_N)
#           其中 n_N 为目标层节点，n₁..n_{N-1} 为中间层节点
#   rev[c]: 到达目标节点 c 的全部路径，每条路径 = N 元组 (m₁, ..., m_{N-1}, a)
#           其中 a 为源节点，m₁..m_{N-1} 为中间层节点 (从近 c 端到近 a 端)
#   fwd_w[a]: fwd[a] 中每条路径对应的 weight
#   rev_w[c]: rev[c] 中每条路径对应的 weight

struct LayerMap{N,D₁,D₂,T}
    fwd::Vector{Vector{NTuple{N,Int64}}}
    rev::Vector{Vector{NTuple{N,Int64}}}
    fwd_w::Vector{Vector{T}}
    rev_w::Vector{Vector{T}}
end

function Base.:(==)(bm1::LayerMap{N,D₁,D₂,T}, bm2::LayerMap{N,D₁,D₂,T}) where {N,D₁,D₂,T}
    return bm1.fwd == bm2.fwd && bm1.rev == bm2.rev &&
           bm1.fwd_w == bm2.fwd_w && bm1.rev_w == bm2.rev_w
end

nsrc(::LayerMap{N,D₁,D₂}) where {N,D₁,D₂} = D₁
ndst(::LayerMap{N,D₁,D₂}) where {N,D₁,D₂} = D₂
Base.size(::LayerMap{N,D₁,D₂}) where {N,D₁,D₂} = (D₁, D₂)
Base.length(bm::LayerMap) = sum(length, bm.fwd)

# getindex(bm, i): 第 i 条路径，返回 (src, n₁, n₂, ..., n_N)
function Base.getindex(bm::LayerMap{N,D₁,D₂,T}, i::Int) where {N,D₁,D₂,T}
    cum = 0
    for a in 1:D₁
        npaths = length(bm.fwd[a])
        if i <= cum + npaths
            p = bm.fwd[a][i - cum]
            return (a, p...)
        end
        cum += npaths
    end
    throw(BoundsError(bm, i))
end

# 迭代：每条路径 (src, n₁, n₂, ..., n_N)
function Base.iterate(bm::LayerMap{N,D₁,D₂,T}, state=(1, 1)) where {N,D₁,D₂,T}
    a, j = state
    while a <= D₁ && j > length(bm.fwd[a])
        a += 1
        j = 1
    end
    a > D₁ && return nothing
    p = bm.fwd[a][j]
    return ((a, p...), (a, j + 1))
end

# getindex(bm, :): 全部路径，返回 Vector{NTuple{N+1, Int64}}
function Base.getindex(bm::LayerMap{N,D₁,D₂,T}, ::Colon) where {N,D₁,D₂,T}
    total = length(bm)
    result = Vector{NTuple{N+1,Int64}}(undef, total)
    k = 1
    for a in 1:D₁
        for p in bm.fwd[a]
            result[k] = (a, p...)
            k += 1
        end
    end
    return result
end

function Base.show(io::IO, bm::LayerMap{N,D₁,D₂,T}) where {N,D₁,D₂,T}
    edges = sum(length, bm.fwd)
    print(io, "LayerMap{$N,$T}($D₁ ↔ $D₂, $edges paths)")
end

# 从边函数 edges(a) -> [子节点索引] 直接构建 1 阶 LayerMap，不经过 Dict
function LayerMap(edges::Function, nA::Int, nB::Int)
    fwd = Vector{Vector{NTuple{1,Int64}}}(undef, nA)
    rev = Vector{Vector{NTuple{1,Int64}}}(undef, nB)
    for b in 1:nB
        rev[b] = NTuple{1,Int64}[]
    end
    for a in 1:nA
        cs = sort!(edges(a))
        fwd[a] = NTuple{1,Int64}[(c,) for c in cs]
        for (c,) in fwd[a]
            push!(rev[c], (a,))
        end
    end
    fwd_w = [fill(1.0, length(fwd[a])) for a in 1:nA]
    rev_w = [fill(1.0, length(rev[b])) for b in 1:nB]
    return LayerMap{1,nA,nB,Number}(fwd, rev, fwd_w, rev_w)
end

# 从正向邻接字典构建 1 阶 LayerMap (兼容旧接口)
function LayerMap(fwd_dict::Dict{Int64,Vector{Int64}}, nA::Int, nB::Int)
    fwd = Vector{Vector{NTuple{1,Int64}}}(undef, nA)
    for a in 1:nA
        cs = get(fwd_dict, Int64(a), Int64[])
        fwd[a] = NTuple{1,Int64}[(c,) for c in sort!(cs)]
    end
    rev = Vector{Vector{NTuple{1,Int64}}}(undef, nB)
    for b in 1:nB
        rev[b] = NTuple{1,Int64}[]
    end
    for a in 1:nA
        for (c,) in fwd[a]
            push!(rev[c], (a,))
        end
    end
    fwd_w = [fill(1.0, length(fwd[a])) for a in 1:nA]
    rev_w = [fill(1.0, length(rev[b])) for b in 1:nB]
    return LayerMap{1,nA,nB,Number}(fwd, rev, fwd_w, rev_w)
end

# 关系复合: R₁: A→B (N₁阶), R₂: B→C (N₂阶) → R₁∘R₂: A→C (N₁+N₂阶)
function compose(r1::LayerMap{N₁,D₁,DM,T}, r2::LayerMap{N₂,DM,D₂,T}) where {N₁,N₂,D₁,DM,D₂,T}
    N = N₁ + N₂

    fwd_new = Vector{Vector{NTuple{N,Int64}}}(undef, D₁)
    fwd_w_new = Vector{Vector{T}}(undef, D₁)
    for a in 1:D₁
        paths = NTuple{N,Int64}[]
        weights = T[]
        for (pi1, w1) in enumerate(r1.fwd_w[a])
            p1 = r1.fwd[a][pi1]
            b = p1[N₁]
            b <= DM || continue
            for (pi2, w2) in enumerate(r2.fwd_w[b])
                p2 = r2.fwd[b][pi2]
                push!(paths, (p1..., p2...))
                push!(weights, w1 * w2)
            end
        end
        fwd_new[a] = paths
        fwd_w_new[a] = weights
    end

    rev_new = Vector{Vector{NTuple{N,Int64}}}(undef, D₂)
    rev_w_new = Vector{Vector{T}}(undef, D₂)
    for c in 1:D₂
        paths = NTuple{N,Int64}[]
        weights = T[]
        for (pi2, w2) in enumerate(r2.rev_w[c])
            p2 = r2.rev[c][pi2]
            b = p2[N₂]
            b <= DM || continue
            for (pi1, w1) in enumerate(r1.rev_w[b])
                p1 = r1.rev[b][pi1]
                push!(paths, (p2..., p1...))
                push!(weights, w2 * w1)
            end
        end
        rev_new[c] = paths
        rev_w_new[c] = weights
    end

    return LayerMap{N,D₁,D₂,T}(fwd_new, rev_new, fwd_w_new, rev_w_new)
end

# 三元复合: R₁: A→B, R₂: B→C, R₃: C→D → R₁∘R₂∘R₃: A→D
function compose(r1::LayerMap{N₁,D₁,DM₁,T}, r2::LayerMap{N₂,DM₁,DM₂,T},
                 r3::LayerMap{N₃,DM₂,D₂,T}) where {N₁,N₂,N₃,D₁,DM₁,DM₂,D₂,T}
    return compose(compose(r1, r2), r3)
end

# 从 DirectedNode 向量构建 1 阶 LayerMap，从 DirectedEdge 提取 weight
function LayerMap(src::Vector{<:DirectedNode}, dst::Vector{<:DirectedNode})
    nA = length(src)
    nB = length(dst)
    node2idx = Dict{Any,Int64}(n => Int64(i) for (i, n) in enumerate(dst))

    fwd = Vector{Vector{NTuple{1,Int64}}}(undef, nA)
    fwd_w = Vector{Vector{Number}}(undef, nA)
    rev = Vector{Vector{NTuple{1,Int64}}}(undef, nB)
    rev_w = Vector{Vector{Number}}(undef, nB)
    for b in 1:nB
        rev[b] = NTuple{1,Int64}[]
        rev_w[b] = Number[]
    end

    for a in 1:nA
        # (target_idx, weight) pairs
        pairs = Tuple{Int64,Number}[]
        for e in src[a].out_edges
            for r in _resolve_forward(e.to)
                ci = get(node2idx, r, nothing)
                ci !== nothing && push!(pairs, (ci, e.weight))
            end
        end
        sort!(pairs, by = x -> x[1])
        uniq_fwd = NTuple{1,Int64}[]
        uniq_fwd_w = Number[]
        for (ci, w) in pairs
            if !isempty(uniq_fwd) && uniq_fwd[end][1] == ci
                uniq_fwd_w[end] += w
            else
                push!(uniq_fwd, (ci,))
                push!(uniq_fwd_w, w)
            end
        end
        fwd[a] = uniq_fwd
        fwd_w[a] = uniq_fwd_w
        for (ci,) in uniq_fwd
            push!(rev[ci], (a,))
        end
    end

    # Build rev_w from fwd data
    for a in 1:nA
        for (pi, (ci,)) in enumerate(fwd[a])
            push!(rev_w[ci], fwd_w[a][pi])
        end
    end

    return LayerMap{1,nA,nB,Number}(fwd, rev, fwd_w, rev_w)
end
