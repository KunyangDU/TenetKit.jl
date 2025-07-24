using AbstractTrees

abstract type AbstractTreeNode end

mutable struct InteractionTreeNode <: AbstractTreeNode

    Opr::Union{Nothing,AbstractLocalOperator}
    Leave::Union{Nothing,InteractionTreeLeave}
    parent::Union{Nothing,InteractionTreeNode}
    children::Vector{InteractionTreeNode}

    function InteractionTreeNode(
        Opr::Union{Nothing,AbstractLocalOperator},
        parent::InteractionTreeNode,
        children::Vector{InteractionTreeNode}=InteractionTreeNode[],
    )
    return new(Opr,nothing,parent,children)
    end

    function InteractionTreeNode(
        Opr::Union{Nothing,AbstractLocalOperator},
        children::Vector{InteractionTreeNode}=InteractionTreeNode[],
    )
    return new(Opr,nothing,nothing,children)
    end

    InteractionTreeNode() = InteractionTreeNode(IdentityOperator(0))
end

AbstractTrees.nodevalue(node::InteractionTreeNode) = node.Opr
AbstractTrees.parent(node::AbstractTreeNode) = node.parent
AbstractTrees.children(node::AbstractTreeNode) = node.children
AbstractTrees.ParentLinks(::Type{AbstractTreeNode}) = StoredParents()
AbstractTrees.ChildIndexing(::Type{AbstractTreeNode}) = IndexedChildren()
AbstractTrees.NodeType(::Type{AbstractTreeNode}) = HasNodeType()
AbstractTrees.nodetype(::T) where T <: AbstractTreeNode = T

function addchild!(node::AbstractTreeNode, child::AbstractTreeNode)
    isnothing(child.parent) ? child.parent = node : @assert child.parent == node
    push!(node.children, child)
    return nothing
end

function addchild!(node::T, Opr::AbstractLocalOperator) where T <: AbstractTreeNode
    addchild!(node,T(Opr))
    return nothing
end

function cutparent!(node::AbstractTreeNode)
    node.parent = nothing
    return node
end


function Base.show(io::IO,Root::AbstractTreeNode)
    print_tree(Root;maxdepth = 16)
    return nothing
end


struct InteractionTree{N}
    Root::InteractionTreeNode
    function InteractionTree(child::InteractionTreeNode...)
         N = length(child)
         
         Root = InteractionTreeNode(nothing)
         for i in child
              addchild!(Root, i)
         end
         return new{N}(Root)
    end

    function InteractionTree{N}() where N
         
        Root = InteractionTreeNode(nothing)
        for i in 1:N
             addchild!(Root, InteractionTreeNode(IdentityOperator(0)))
        end
        return new{N}(Root)
   end

   InteractionTree() = InteractionTree(InteractionTreeNode(IdentityOperator(0)))
end

