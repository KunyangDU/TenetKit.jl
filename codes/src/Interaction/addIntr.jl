# ========================= addIntr! for InteractionGraph =========================
# 接口与旧 addIntr! 一致，将 Root::AbstractTreeNode 替换为 ig::InteractionGraph。
# 每条相互作用被包装为 InteractionTunnel 存入 ig.tunnel，
# 调用 build_sparse_mpo(ig) 时才建图 + 生成 SparseMPO。

# -- 标量形式 (N=1) --
function addIntr!(ig::InteractionGraph,
    A::AbstractTensorMap,
    site::Int64,
    name::String,
    fermionic::Bool,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength ≈ 0 && return nothing
    tunnel = InteractionTunnel((A,), (site,), (name,), (fermionic,), strength, Z, ig.L, LocalOperator)
    push!(ig.tunnel, tunnel)
    ig.graph = nothing
    return nothing
end

# -- Tuple 形式 (任意 N) --
function addIntr!(ig::InteractionGraph,
    A::NTuple{N,AbstractTensorMap},
    site::NTuple{N,Int64},
    name::NTuple{N,String},
    fermionic::NTuple{N,Bool},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap}) where N
    strength ≈ 0 && return nothing
    tunnel = InteractionTunnel(A, site, name, fermionic, strength, Z, ig.L, LocalOperator)
    push!(ig.tunnel, tunnel)
    ig.graph = nothing
    return nothing
end
