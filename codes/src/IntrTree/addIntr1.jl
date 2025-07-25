function addIntr1!(Root::AbstractTreeNode,
    A::AbstractTensorMap,site::Int64,name::String,fermionic::Bool,strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    # tempOpr = LocalOperator(A,name,site,strength)
    # addIntr1!(Root,tempOpr,Z)
    Leave = InteractionTreeLeave(A,site,name,fermionic,strength,Z)
    addIntr1!(Root,Leave)
end

function addIntr1!(Root::AbstractTreeNode,Leave::InteractionTreeLeave)
    A = LocalOperator(Leave)
    Z = Leave.Z
    current_node = Root
    current_site = 1

    # add the identity
    while current_site < A.site

        if isnothing(Z)
            @assert Leave.fermionic[1] == false
            tempIdOpr = IdentityOperator(getIdTensor(A),current_site)
        else
            @assert Leave.fermionic[1] == true
            tempIdOpr = LocalOperator(Z,"Z",current_site)
        end

        indId = findfirst(x -> isequal(x.A,tempIdOpr),current_node.children)
        if isnothing(indId)
            addchild!(current_node,tempIdOpr)
            current_node = current_node.children[end]
        else
            current_node = current_node.children[indId]
        end

        current_site += 1
    end

    addchild!(current_node, A)
    
    if typeof(Root) <: ObservableTreeNode
        # current_node.children[end].name = Tuple(A.site)
        current_node.children[end].Leave = ObservableTreeLeave(Leave)
    else
        current_node.children[end].Leave = Leave
    end
        # typeof(Root) <: ObservableTreeNode
end

# function addIntr1!(Root::AbstractTreeNode,
#     A::LocalOperator,
#     Z::Union{Nothing,AbstractTensorMap})
#     current_node = Root
#     current_site = 1

#     # add the identity
#     while current_site < A.site

#         if isnothing(Z)
#             tempIdOpr = IdentityOperator(getIdTensor(A),current_site)
#         else
#             tempIdOpr = LocalOperator(Z,"Z",current_site,1)
#         end

#         indId = findfirst(x -> isequal(x.A,tempIdOpr),current_node.children)
#         if isnothing(indId)
#             addchild!(current_node,tempIdOpr)
#             current_node = current_node.children[end]
#         else
#             current_node = current_node.children[indId]
#         end

#         current_site += 1
#     end

#     # add the onsite A 
#     addchild!(current_node, A)
#     # current_node.children[end].A.strength = strength
#     typeof(Root) <: ObservableTreeNode && (current_node.children[end].name = Tuple(A.site))
# end