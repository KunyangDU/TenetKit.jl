
mutable struct InteractionTunnel{L,T,N} <: AbstractTunnel{L,T}
    A::NTuple{N,T}
    fermionic::NTuple{N,Bool}
    Z::Union{Nothing,AbstractTensorMap}
    strength::Float64
    function InteractionTunnel(
        As::NTuple{N,AbstractTensorMap},
        sites::NTuple{N,Int64},
        names::NTuple{N,String},
        fermionic::NTuple{N,Bool},
        strength::Number,
        Z::Union{Nothing,AbstractTensorMap},
        L::Int64,T::Type = LocalOperator
    ) where N
        # 按 site 排序，使 getindex 利用递增性质避免 findfirst
        perm = sortperm([sites...])
        ops = ntuple(i -> T(As[perm[i]], names[perm[i]], sites[perm[i]]), N)
        ferm = ntuple(i -> fermionic[perm[i]], N)
        new{L,T,N}(ops, ferm, Z, Float64(strength))
    end
    function InteractionTunnel{L,T}(A::NTuple{N,T}, fermionic::NTuple{N,Bool}, Z::Union{Nothing,AbstractTensorMap}, strength::Float64) where {L,T,N}
        return new{L,T,N}(A,fermionic,Z,strength)
    end

    function InteractionTunnel(
        A::AbstractTensorMap,
        site::Int64,
        name::String,
        fermionic::Bool,
        strength::Number,
        Z::Union{Nothing,AbstractTensorMap},
        L::Int64,T::Type = LocalOperator
    )
        return new{L,T,1}((T(A,name,site),),(fermionic,),Z,strength)
    end

    function InteractionTunnel(
        A::NTuple{N,T},
        fermionic::NTuple{N,Bool},
        Z::Union{Nothing,AbstractTensorMap},
        strength::Float64, L::Int64,T′::Type = T
    ) where {N,T}
        @assert T <: T′
        return new{L,T′,N}(A,fermionic,Z,strength)
    end
end

function Base.getindex(obj::InteractionTunnel{L,<:LocalOperator,N}, i::Int64) where {L,N}
    # A 按 site 递增排序，单次线性扫描
    @inbounds for idx in 1:N
        site = obj.A[idx].site
        if site == i
            return obj.A[idx]
        elseif site > i
            isnothing(obj.Z) && return IdentityOperator(i)
            return isfermionic(obj, i) ? LocalOperator(obj.Z, "Z", i) : IdentityOperator(i)
        end
    end
    return IdentityOperator(i)
end

mutable struct CompositeInteractionTunnel{L,T,N} <: AbstractTunnel{L,T}
    A::NTuple{N,InteractionTunnel{L}}
    strength::Float64
end

Base.getindex(obj::CompositeInteractionTunnel, i::Int64) = CompositeLocalOperator([a[i] for a in obj.A])
composite(A::InteractionTunnel{L,T}, B::InteractionTunnel{L,T}) where {L,T <: LocalOperator} = CompositeInteractionTunnel{L,CompositeLocalOperator{2},2}(NTuple{2,InteractionTunnel{L}}((A,B)), A.strength * B.strength)


# site i 处的 JW parity = site > i 的费米子数 mod 2
# 从右往左数，每遇到 site > i 的费米子就翻转
function isfermionic(obj::InteractionTunnel, i::Int64)
    parity = false
    for idx in length(obj.A):-1:1
        site = obj.A[idx].site
        if site > i && obj.fermionic[idx]
            parity = !parity
        elseif site <= i
            break
        end
    end
    return parity
end

Base.getindex(obj::AbstractTunnel, r::UnitRange{Int64}) = [obj[i] for i in r]
Base.length(::AbstractTunnel{L}) where L = L

_site(A::InteractionTunnel) = map(x -> x.site,A.A)
Base.:*(A::Number, B::InteractionTunnel) = (B′ = deepcopy(B); B′.strength *= A; return B′)


# ============================================================
# InteractionTunnelSegment: tunnel[from:to] 的轻量引用，支持 Dict key
# ============================================================
struct InteractionTunnelSegment{L,T}
    tunnel::InteractionTunnel{L,T}
    from::Int64
    to::Int64
end

# isequal 三步走（利用稀疏性，不逐位比较）：
#   1. Z 算符必须相同
#   2. 初始 JW parity 必须相同（from 右侧费米子数的奇偶）
#   3. 段内显式算符按相对 offset 一一对应
# 三步都通过 → 空位必然一致，段相等

function Base.isequal(a::InteractionTunnelSegment{L,T}, b::InteractionTunnelSegment{L,T}) where {L,T}
    len_a = a.to - a.from + 1
    len_b = b.to - b.from + 1
    len_a == len_b || return false
    len_a == 0 && return true

    A_a, Na = a.tunnel.A, length(a.tunnel.A)
    A_b, Nb = b.tunnel.A, length(b.tunnel.A)

    # Step 1: JW parity at `from`（site > from 的费米子数 mod 2）
    pa = isfermionic(a.tunnel, a.from)
    pb = isfermionic(b.tunnel, b.from)
    pa == pb || return false

    # parity == true 时空位才会产生 Z 算符，此时才需比较 Z
    pa && !isequal(a.tunnel.Z, b.tunnel.Z) && return false

    # 游标定位到段起点（Step 3 用）
    ca = 1; while ca <= Na && A_a[ca].site < a.from; ca += 1; end
    cb = 1; while cb <= Nb && A_b[cb].site < b.from; cb += 1; end

    # Step 3: 段内显式算符按相对 offset 对齐比较
    while ca <= Na && A_a[ca].site <= a.to && cb <= Nb && A_b[cb].site <= b.to
        off_a = A_a[ca].site - a.from
        off_b = A_b[cb].site - b.from
        if off_a == off_b
            isequal(A_a[ca], A_b[cb]) || return false
            ca += 1; cb += 1
        elseif off_a < off_b
            return false   # a 在 offset 有算符，b 没有
        else
            return false   # b 在 offset 有算符，a 没有
        end
    end
    # 检查剩余：任一段还有段内算符 → 不相等
    while ca <= Na && A_a[ca].site <= a.to; return false; end
    while cb <= Nb && A_b[cb].site <= b.to; return false; end
    return true
end

# hash: 逐位带 parity，空位直接 hash 标记值，不分配算符对象
function Base.hash(s::InteractionTunnelSegment{L,T}, h::UInt) where {L,T}
    len = s.to - s.from + 1
    h = hash(len, h)
    len == 0 && return h

    A, ferm, Z = s.tunnel.A, s.tunnel.fermionic, s.tunnel.Z
    N = length(A)
    c = 1; while c <= N && A[c].site < s.from; c += 1; end

    parity = 0; for j in c:N; ferm[j] && (parity ⊻= 1); end
    if c <= N && A[c].site == s.from; parity ⊻= ferm[c]; end

    for offset in 0:len-1
        pos = s.from + offset
        if c <= N && A[c].site == pos
            h = hash(A[c], h)
            ferm[c] && (parity ⊻= 1)
            c += 1
        elseif parity == 1 && !isnothing(Z)
            h = hash(Z, h)
            h = hash(pos, h)
        else
            h = hash(pos, h)       # Identity，只 hash 位置
        end
    end
    return h
end