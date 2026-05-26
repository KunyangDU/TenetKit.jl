# LayerMap{N,D₁,D₂}: N 阶层间映射，编码 A → B 经过 N 步的路径关系
#   D₁ = 入射维数 (|A|), D₂ = 出射维数 (|B|)
#   N=1: 相邻层直连映射，路径元组长度 1
#   N≥2: compose 产生的多阶层间映射，路径保留全部中间节点
#
#   fwd[a]: 从源节点 a 出发的全部路径，每条路径 = N 元组 (n₁, n₂, ..., n_N)
#           其中 n_N 为目标层节点，n₁..n_{N-1} 为中间层节点
#   rev[c]: 到达目标节点 c 的全部路径，每条路径 = N 元组 (m₁, ..., m_{N-1}, a)
#           其中 a 为源节点，m₁..m_{N-1} 为中间层节点 (从近 c 端到近 a 端)

struct LayerMap{N,D₁,D₂}
    fwd::Vector{Vector{NTuple{N,Int64}}}
    rev::Vector{Vector{NTuple{N,Int64}}}
end

nsrc(::LayerMap{N,D₁,D₂}) where {N,D₁,D₂} = D₁
ndst(::LayerMap{N,D₁,D₂}) where {N,D₁,D₂} = D₂
Base.size(::LayerMap{N,D₁,D₂}) where {N,D₁,D₂} = (D₁, D₂)
Base.length(bm::LayerMap) = sum(length, bm.fwd)

# getindex(bm, i): 第 i 条路径，返回 (src, n₁, n₂, ..., n_N)
function Base.getindex(bm::LayerMap{N,D₁,D₂}, i::Int) where {N,D₁,D₂}
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
function Base.iterate(bm::LayerMap{N,D₁,D₂}, state=(1, 1)) where {N,D₁,D₂}
    a, j = state
    while a <= D₁ && j > length(bm.fwd[a])
        a += 1
        j = 1
    end
    a > D₁ && return nothing
    p = bm.fwd[a][j]
    return ((a, p...), (a, j + 1))
end

function Base.show(io::IO, bm::LayerMap{N,D₁,D₂}) where {N,D₁,D₂}
    edges = sum(length, bm.fwd)
    print(io, "LayerMap{$N}($D₁ ↔ $D₂, $edges paths)")
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
    return LayerMap{1,nA,nB}(fwd, rev)
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
    return LayerMap{1,nA,nB}(fwd, rev)
end

# 关系复合: R₁: A→B (N₁阶), R₂: B→C (N₂阶) → R₁∘R₂: A→C (N₁+N₂阶)
function compose(r1::LayerMap{N₁,D₁,DM}, r2::LayerMap{N₂,DM,D₂}) where {N₁,N₂,D₁,DM,D₂}
    N = N₁ + N₂

    fwd_new = Vector{Vector{NTuple{N,Int64}}}(undef, D₁)
    for a in 1:D₁
        paths = NTuple{N,Int64}[]
        for p1 in r1.fwd[a]           # p1 = (..., b), length N₁
            b = p1[N₁]
            b <= DM || continue
            for p2 in r2.fwd[b]       # p2 = (..., c), length N₂
                push!(paths, (p1..., p2...))
            end
        end
        fwd_new[a] = paths
    end

    rev_new = Vector{Vector{NTuple{N,Int64}}}(undef, D₂)
    for c in 1:D₂
        paths = NTuple{N,Int64}[]
        for p2 in r2.rev[c]           # p2 = (..., b), length N₂, last=b
            b = p2[N₂]
            b <= DM || continue
            for p1 in r1.rev[b]       # p1 = (..., a), length N₁, last=a
                push!(paths, (p2..., p1...))
            end
        end
        rev_new[c] = paths
    end

    return LayerMap{N,D₁,D₂}(fwd_new, rev_new)
end


function LayerMap(src::Vector, dst::Vector)
    # DirectedNode
    nA = length(src)
    nB = length(dst)
    node2idx = Dict{Any,Int64}(n => Int64(i) for (i, n) in enumerate(dst))
    return LayerMap(nA, nB) do a
        idxs = Int64[]
        for c in src[a].out
            for r in _resolve_forward(c)
                ci = get(node2idx, r, nothing)
                ci !== nothing && push!(idxs, ci)
            end
        end
        return idxs
    end
end
