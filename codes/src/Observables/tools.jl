# function default_val!(node::DirectedNode{T}) where T <: Union{CompositeObservableOperator,ObservableOperator{R₁,R₂}} where {R₁,R₂}
#     leftdelfault_val!(node)
#     rightdelfault_val!(node)
# end

# function leftdelfault_val!(node::DirectedNode{T}) where T <: Union{CompositeObservableOperator,ObservableOperator{R₁,R₂}} where {R₁,R₂}
#     node.val.EnvL = nothing
#     node.val.leftdata = nothing
# end

# function rightdelfault_val!(node::DirectedNode{T}) where T <:  Union{CompositeObservableOperator,ObservableOperator{R₁,R₂}} where {R₁,R₂}
#     node.val.EnvR = nothing
#     node.val.rightdata = nothing
# end
rightdelfault_val!(node::DirectedNode{T}) where T <:  Union{CompositeObservableOperator,ObservableOperator} = rightdelfault_val!(node.val)
leftdelfault_val!(node::DirectedNode{T}) where T <:  Union{CompositeObservableOperator,ObservableOperator} = leftdelfault_val!(node.val)

function default_val!(node::T) where T <: Union{CompositeObservableOperator,ObservableOperator{R₁,R₂}} where {R₁,R₂}
    leftdelfault_val!(node)
    rightdelfault_val!(node)
end

function leftdelfault_val!(node::T) where T <: Union{CompositeObservableOperator,ObservableOperator{R₁,R₂}} where {R₁,R₂}
    node.EnvL = nothing
    node.leftdata = nothing
end

function rightdelfault_val!(node::T) where T <:  Union{CompositeObservableOperator,ObservableOperator{R₁,R₂}} where {R₁,R₂}
    node.EnvR = nothing
    node.rightdata = nothing
end

isdefault(node::T) where T <: Union{ObservableOperator{R₁,R₂}, CompositeObservableOperator} where {R₁,R₂} = (isleftdefault(node) && isrightdefault(node))
# isrightdefault(node::T) where T <: Union{ObservableOperator{R₁,R₂}, CompositeObservableOperator} where {R₁,R₂} = isnothing(node.EnvL) && isnothing(node.leftdata)
# isleftdefault(node::T) where T <: Union{ObservableOperator{R₁,R₂}, CompositeObservableOperator} where {R₁,R₂} = isnothing(node.EnvR) && isnothing(node.rightdata)
isleftdefault(node::T) where T <: Union{ObservableOperator{R₁,R₂}, CompositeObservableOperator} where {R₁,R₂} = (isnothing(node.EnvL) && isnothing(node.leftdata))
isrightdefault(node::T) where T <: Union{ObservableOperator{R₁,R₂}, CompositeObservableOperator} where {R₁,R₂} = (isnothing(node.EnvR) && isnothing(node.rightdata))

_lr_merge(left::Dict,right::Dict) = map(x -> _lr_merge(left[x],right[x]), ["name","site"])

contract(EnvL::LeftEnvironmentTensor{2},EnvR::RightEnvironmentTensor{2}) = @tensor EnvL.A[1,2] * EnvR.A[2,1]
contract(EnvL::LeftEnvironmentTensor{3},EnvR::RightEnvironmentTensor{3}) = @tensor EnvL.A[1,2,3] * EnvR.A[3,2,1]

function contract(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    # opt=true: greedy optimizer contracts small (d²) local operators and EnvL (D²) before
    # ever pairing objt with objb.  Without it the default left-to-right order reaches
    # objt×objb — which share only d-sized bonds — producing a D⁴ intermediate
    # (~1.6 GB at D=100).  opt=true keeps every intermediate at O(d²D²).
    # @tensor opt=true x[-1;-2] ≔ ht.A[2,6] * objt.A[1,3,-2,2] * hb.A[5,1] * objb.A[-1,6,5,4] * EnvL.A[4,3]
    @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[2,3,-2,1] * hb.A[6,2] * objb.A[-1,5,6,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function contract(::IdentityOperator{1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    # @tensor opt=true x[-1;-2] ≔ objt.A[1,3,-2,2] * hb.A[5,1] * objb.A[-1,2,5,4] * EnvL.A[4,3]
    @tensor x[-1;-2] ≔ objt.A[2,3,-2,5] * hb.A[6,2] * objb.A[-1,5,6,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function contract(::IdentityOperator{1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    # Even with only 3 tensors, the default order contracts objt×objb first (D⁴).
    # @tensor opt=true x[-1;-2] ≔ objt.A[1,3,-2,2] * objb.A[-1,2,1,4] * EnvL.A[4,3]
    @tensor x[-1;-2] ≔  objt.A[6,3,-2,5] * objb.A[-1,5,6,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function contract(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    # @tensor opt=true x[-2;-1] ≔ ht.A[2,5] * objt.A[1,3,-2,2] * objb.A[-1,5,1,4] * EnvL.A[4,3]
    @tensor x[-1;-2] ≔ ht.A[1,5] * objt.A[6,3,-2,1] * objb.A[-1,5,6,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function contract(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor x[-2;-1] ≔ ht.A[1,5] * objt.A[2,-2,3,1] * hb.A[6,2] * objb.A[4,5,6,-1] * EnvR.A[3,4]
    return RightEnvironmentTensor(x)
end

function contract(::IdentityOperator{1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor x[-2;-1] ≔ objt.A[2,-2,3,5] * hb.A[6,2] * objb.A[4,5,6,-1] * EnvR.A[3,4]
    return RightEnvironmentTensor(x)
end

function contract(::IdentityOperator{1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor x[-2;-1] ≔  objt.A[6,-2,3,5] * objb.A[4,5,6,-1] * EnvR.A[3,4]
    return RightEnvironmentTensor(x)
end

function contract(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvR::RightEnvironmentTensor{2})
    @tensor x[-2;-1] ≔ ht.A[1,5] * objt.A[6,-2,3,1] * objb.A[4,5,6,-1] * EnvR.A[3,4]
    return RightEnvironmentTensor(x)
end

dictsize(d::Dict) = sum(v -> v isa Dict ? dictsize(v) : 1, values(d); init=0)

function deepmerge!(d1::Dict, d2::Dict)
    for (k, v2) in d2
        if haskey(d1, k) && d1[k] isa Dict && v2 isa Dict
            deepmerge!(d1[k], v2)
        else
            d1[k] = v2
        end
    end
    return d1
end

function leftmergedata!(w::ObservableWeight,n::ObservableOperator)
    push!(w.leftdata["name"], n.name)
    push!(w.leftdata["site"], n.site)
end

function leftmergedata!(w::ObservableWeight,n::CompositeObservableOperator{2})
    for i in eachindex(n.A)
        n.isstring[i] && continue
        # !isempty(val.leftdata["site"][i]) && val.leftdata["site"][i][end] == val.A[i].site && continue
        push!(w.leftdata["name"][i], n.A[i].name)
        push!(w.leftdata["site"][i], n.A[i].site)
    end
end

function rightmergedata!(w::ObservableWeight,n::ObservableOperator)
    push!(w.rightdata["name"], n.name)
    push!(w.rightdata["site"], n.site)
end

function rightmergedata!(w::ObservableWeight,n::CompositeObservableOperator{2})
    for i in eachindex(n.A)
        n.isstring[i] && continue
        # !isempty(val.rightdata["site"][i]) && val.rightdata["site"][i][end] == val.A[i].site && continue
        push!(w.rightdata["name"][i], n.A[i].name)
        push!(w.rightdata["site"][i], n.A[i].site)
    end
end

function leftmergedata!(val::ObservableOperator)
    push!(val.leftdata["name"], val.name)
    push!(val.leftdata["site"], val.site)
end

function rightmergedata!(val::ObservableOperator)
    push!(val.rightdata["name"], val.name)
    push!(val.rightdata["site"], val.site)
end

function leftmergedata!(val::CompositeObservableOperator{2})
    for i in eachindex(val.A)
        val.isstring[i] && continue
        # !isempty(val.leftdata["site"][i]) && val.leftdata["site"][i][end] == val.A[i].site && continue
        push!(val.leftdata["name"][i], val.A[i].name)
        push!(val.leftdata["site"][i], val.A[i].site)
    end
end

function rightmergedata!(val::CompositeObservableOperator{2})
    for i in eachindex(val.A)
        val.isstring[i] && continue
        # !isempty(val.rightdata["site"][i]) && val.rightdata["site"][i][end] == val.A[i].site && continue
        push!(val.rightdata["name"][i], val.A[i].name)
        push!(val.rightdata["site"][i], val.A[i].site)
    end
end

_site(val::ObservableOperator) = val.site 
_site(val::CompositeObservableOperator) = (sites = map(x -> x.site,val.A); @assert reduce(==, sites); sites[1])


_lr_merge(left::Vector{T},right::Vector{T}) where T <: Union{Int64,String} = tuple(left..., reverse(right)...)
_lr_merge(left::Vector{Vector{T}},right::Vector{Vector{T}}) where T <: Union{Int64,String} = (@assert length(left) == length(right) ; Tuple(map(x -> _lr_merge(left[x],right[x]), eachindex(left))))

_calObs_left_contract(val::ObservableOperator, obj::T) where T <: Union{MPSTensor, DenseMPOTensor} = contract(obj, isnothing(val.A) ? IdentityOperator(val.site) : LocalOperator(val.A, val.name, val.site), obj', val.EnvL)
_calObs_right_contract(val::ObservableOperator, obj::T) where T <: Union{MPSTensor, DenseMPOTensor} = contract(obj, isnothing(val.A) ? IdentityOperator(val.site) : LocalOperator(val.A, val.name, val.site), obj', val.EnvR)

_calObs_left_contract(val::CompositeObservableOperator{2}, obj::T) where T <: Union{MPSTensor, DenseMPOTensor} = contract(val.A[1], obj, val.A[2], obj', val.EnvL)
_calObs_right_contract(val::CompositeObservableOperator{2}, obj::T) where T <: Union{MPSTensor, DenseMPOTensor} = contract(val.A[1], obj, val.A[2], obj', val.EnvR)

_calObs_left_contract!(w::ObservableWeight, val::ObservableOperator, obj::T) where T <: Union{MPSTensor, DenseMPOTensor} = (w.EnvL = contract(obj, isnothing(val.A) ? IdentityOperator(val.site) : LocalOperator(val.A, val.name, val.site), obj', w.EnvL) * w.strength)
_calObs_right_contract!(w::ObservableWeight, val::ObservableOperator, obj::T) where T <: Union{MPSTensor, DenseMPOTensor} = (w.EnvR = contract(obj, isnothing(val.A) ? IdentityOperator(val.site) : LocalOperator(val.A, val.name, val.site), obj', w.EnvR) * w.strength)

_calObs_left_contract!(w::ObservableWeight, val::CompositeObservableOperator{2}, obj::T) where T <: Union{MPSTensor, DenseMPOTensor} = (w.EnvL = contract(val.A[1], obj, val.A[2], obj', w.EnvL) * w.strength)
_calObs_right_contract!(w::ObservableWeight, val::CompositeObservableOperator{2}, obj::T) where T <: Union{MPSTensor, DenseMPOTensor} = (w.EnvR = contract(val.A[1], obj, val.A[2], obj', w.EnvR) * w.strength)

_leftsite(val::ObservableOperator) = _endsite(val.leftdata["site"])
_rightsite(val::ObservableOperator) = _endsite(val.rightdata["site"])
_endsite(d::Vector{Int64}) = isempty(d) ? NaN : d[end] 
_leftsite(val::CompositeObservableOperator) = (site = 0; map(x -> (!isempty(x) && (site = max(site,_endsite(x)))), val.leftdata["site"]); return site)
_rightsite(val::CompositeObservableOperator) = (site = Inf; map(x -> (!isempty(x) && (site = Int64(min(site,_endsite(x))))), val.rightdata["site"]); return site)
# _leftsite(w::ObservableWeight) = _endsite(w.leftdata["site"])
# _rightsite(w::ObservableWeight) = _endsite(w.rightdata["site"])

function observe(ig::InteractionGraph{L,LocalOperator}) where L
    # @assert isnothing(ig.graph)
    obs = ObservableGraph(L)
    obs.tunnel = observe.(ig.tunnel)
    return obs
end

observe(t::InteractionTunnel{L, LocalOperator, N}) where {L,N} = InteractionTunnel(map(x -> ObservableOperator(x, false),t.A), t.fermionic, t.Z, t.strength, L, ObservableOperator)

hasLR(node::DirectedNode) = !isleftdefault(node.val), !isrightdefault(node.val)
hasLR(nodeL::DirectedNode, nodeR::DirectedNode) = hasLR(nodeL)..., hasLR(nodeR)...
hasLR(w::DirectedEdge) = !isleftdefault(w), !isrightdefault(w)

