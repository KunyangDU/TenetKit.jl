mutable struct ObservableTreeNode <: AbstractTreeNode
    Opr::Union{Nothing,AbstractLocalOperator}
    Env::Union{Nothing,AbstractEnvironmentTensor}
    parent::Union{Nothing,ObservableTreeNode}
    children::Vector{ObservableTreeNode}
    value::Union{Nothing,Number}
    name::Union{Nothing,Tuple}

    function ObservableTreeNode(
        Opr::Union{Nothing,AbstractLocalOperator},
        parent::ObservableTreeNode,
        children::Vector{ObservableTreeNode}=ObservableTreeNode[],
    )
    return new(Opr,nothing,parent,children,NaN,nothing)
    end

    function ObservableTreeNode(
        Opr::Union{Nothing,AbstractLocalOperator},
        children::Vector{ObservableTreeNode}=ObservableTreeNode[],
    )
    return new(Opr,nothing,nothing,children,NaN,nothing)
    end

    ObservableTreeNode() = ObservableTreeNode(IdentityOperator(0))
end

