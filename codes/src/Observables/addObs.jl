
# -- 标量形式 (N=1) --
function addObs!(ig::InteractionGraph,
    A::AbstractTensorMap,
    site::Int64,
    name::String,
    fermionic::Bool,
    Z::Union{Nothing,AbstractTensorMap})
    tunnel = InteractionTunnel((A,), (site,), (name,), (fermionic,), 1.0, Z, ig.L, ObservableOperator)
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
    Z::Union{Nothing,AbstractTensorMap}) where N
    tunnel = InteractionTunnel(A, site, name, fermionic, 1.0, Z, ig.L, ObservableOperator)
    push!(ig.tunnel, tunnel)
    ig.graph = nothing
    return nothing
end
