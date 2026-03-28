function addString!(
    Root::AbstractTreeNode,
    As::Vector,
    sites::Vector,
    names::Vector,
    fermionic::Vector,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap}
    )
    Leave = InteractionTreeLeave(As,sites,names,fermionic,strength,Z)
    addString!(Root,Leave)
end

function addString!(Root::AbstractTreeNode,Leave::InteractionTreeLeave{L}) where L

    Oprs = map(s -> LocalOperator(Leave.A[s],Leave.name[s],s), 1:L)
    current_node = Root

    for s in 1:L-1

        indId = findfirst(x -> isequal(x.A,Oprs[s]),current_node.children)

        if isnothing(indId)
            addchild!(current_node,Oprs[s])
            current_node = current_node.children[end]
        else
            current_node = current_node.children[indId]
        end

    end

    addchild!(current_node,Oprs[end])

    if typeof(Root) <: ObservableTreeNode
        current_node.children[end].Leave = ObservableTreeLeave(Leave)
    else
        current_node.children[end].Leave = Leave
    end
end