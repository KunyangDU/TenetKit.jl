
# -- 标量形式 (N=1) --
function addObs!(ig::InteractionGraph,
    A::AbstractTensorMap,
    site::Int64,
    name::String,
    fermionic::Bool,
    Z::Union{Nothing,AbstractTensorMap}, strength::Number = 1.0)
    strength ≈ 0 && return nothing
    tunnel = InteractionTunnel((A,), (site,), (name,), (fermionic,), strength, Z, ig.L, LocalOperator)
    push!(ig.tunnel, tunnel)
    ig.graph = nothing
    return nothing
end

# -- Tuple 形式 (任意 N) --
function addObs!(ig::InteractionGraph,
    A::NTuple{N,AbstractTensorMap},
    site::NTuple{N,Int64},
    name::NTuple{N,String},
    fermionic::NTuple{N,Bool},
    Z::Union{Nothing,AbstractTensorMap}, strength::Number = 1.0) where N
    strength ≈ 0 && return nothing
    tunnel = InteractionTunnel(A, site, name, fermionic, strength, Z, ig.L, LocalOperator)
    push!(ig.tunnel, tunnel)
    ig.graph = nothing
    return nothing
end
