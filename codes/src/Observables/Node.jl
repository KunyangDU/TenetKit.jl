mutable struct ObservableTreeLeave{N}
    site::Union{Vector,NTuple{N,Tuple}}
    name::Union{Vector,NTuple{N,Tuple}}
    value::Number
    function ObservableTreeLeave(A::InteractionTreeLeave, i::Int64 = 1, N::Int64 = 1)
        site = Vector(undef,N)
        name = Vector(undef,N)
        site[i] = A.site
        name[i] = A.name
        return new{N}(site,name,NaN)
    end
end

# Global counter for unique node ids — all Observable tree nodes share this.
const _NEXT_NODE_ID = Ref{UInt64}(0)
_next_id() = (_NEXT_NODE_ID[] += 1)

mutable struct ObservableTreeNode <: AbstractObservableTreeNode
    id::UInt64
    A::Union{Nothing,AbstractLocalOperator}
    parent::Union{Nothing,ObservableTreeNode}
    children::Vector{ObservableTreeNode}
    Leave::Union{Nothing,ObservableTreeLeave}
    cachedict::Union{Nothing,CachedDict}

    function ObservableTreeNode(
        A::Union{Nothing,AbstractLocalOperator},
        parent::ObservableTreeNode,
        children::Vector{ObservableTreeNode}=ObservableTreeNode[],
    )
    return new(_next_id(), A, parent, children, nothing, nothing)
    end

    function ObservableTreeNode(
        A::Union{Nothing,AbstractLocalOperator},
        children::Vector{ObservableTreeNode}=ObservableTreeNode[],
    )
    return new(_next_id(), A, nothing, children, nothing, nothing)
    end

    ObservableTreeNode() = new(_next_id(), IdentityOperator(0), nothing, ObservableTreeNode[], nothing, nothing)
end

mutable struct CompositeObservableTreeNode{N} <: AbstractObservableTreeNode
    id::UInt64
    A::NTuple{N,Any}
    parent::Union{Nothing,CompositeObservableTreeNode}
    children::Vector{CompositeObservableTreeNode}
    Leave::Union{Nothing,ObservableTreeLeave}
    cachedict::Union{Nothing,CachedDict}

    function CompositeObservableTreeNode(
        A::NTuple{N,Any},
        parent::CompositeObservableTreeNode,
        children::Vector{CompositeObservableTreeNode}=CompositeObservableTreeNode[],
    ) where N
        return new{N}(_next_id(), A, parent, children, nothing, nothing)
    end

    function CompositeObservableTreeNode(
        A::NTuple{N,Any},
        children::Vector{CompositeObservableTreeNode}=CompositeObservableTreeNode[],
    ) where N
        return new{N}(_next_id(), A, nothing, children, nothing, nothing)
    end
    CompositeObservableTreeNode() = CompositeObservableTreeNode((nothing,nothing))
end
# AbstractTrees.nodevalue(node::CompositeObservableTreeNode) = node.A


