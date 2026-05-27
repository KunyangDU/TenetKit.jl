mutable struct DirectedEdge{T}
    weight::T
    from::DirectedNode
    to::DirectedNode
end

function add_edge!(from::DirectedNode, to::DirectedNode; weight::Float64 = 1.0)
    existing = findfirst(e -> e.to === to, from.out_edges)
    if existing !== nothing
        @assert from.out_edges[existing].weight == weight "bond added twice with non identity coefficient"
        return
    end
    e = DirectedEdge(weight, from, to)
    push!(from.out_edges, e)
    push!(to.in_edges, e)
end

function _remove_edge!(from::DirectedNode, to::DirectedNode)
    idx = findfirst(e -> e.to === to, from.out_edges)
    if idx !== nothing
        e = from.out_edges[idx]
        deleteat!(from.out_edges, idx)
        idx2 = findfirst(e2 -> e2 === e, to.in_edges)
        idx2 !== nothing && deleteat!(to.in_edges, idx2)
    end
end