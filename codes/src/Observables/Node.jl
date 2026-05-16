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

mutable struct ObservableTreeNode <: AbstractObservableTreeNode
    A::Union{Nothing,AbstractLocalOperator}
    Env::Union{Nothing,AbstractEnvironmentTensor,String}
    parent::Union{Nothing,ObservableTreeNode}
    children::Vector{ObservableTreeNode}
    Leave::Union{Nothing,ObservableTreeLeave}
    # value::Union{Nothing,Number}
    # name::Union{Nothing,Tuple}
    

    function ObservableTreeNode(
        A::Union{Nothing,AbstractLocalOperator},
        parent::ObservableTreeNode,
        children::Vector{ObservableTreeNode}=ObservableTreeNode[],
    )
    return new(A,nothing,parent,children,nothing)
    end

    function ObservableTreeNode(
        A::Union{Nothing,AbstractLocalOperator},
        children::Vector{ObservableTreeNode}=ObservableTreeNode[],
    )
    return new(A,nothing,nothing,children,nothing)
    end

    ObservableTreeNode() = new(IdentityOperator(0),nothing,nothing,ObservableTreeNode[],nothing)
end

mutable struct CompositeObservableTreeNode{N} <: AbstractObservableTreeNode where {N}
    A::NTuple{N,Any}
    parent::Union{Nothing,CompositeObservableTreeNode}
    children::Vector{CompositeObservableTreeNode}
    Env::Union{Nothing,LeftEnvironmentTensor,String}
    Leave::Union{Nothing,ObservableTreeLeave}

    function CompositeObservableTreeNode(
        A::NTuple{N,Any},
        parent::CompositeObservableTreeNode,
        children::Vector{CompositeObservableTreeNode}=CompositeObservableTreeNode[],
    ) where N
        return new{N}(A,parent,children,nothing,nothing)
    end

    function CompositeObservableTreeNode(
        A::NTuple{N,Any},
        children::Vector{CompositeObservableTreeNode}=CompositeObservableTreeNode[],
    ) where N
        return new{N}(A,nothing,children,nothing,nothing)
    end
    CompositeObservableTreeNode() = CompositeObservableTreeNode((nothing,nothing))
end
# AbstractTrees.nodevalue(node::CompositeObservableTreeNode) = node.A


