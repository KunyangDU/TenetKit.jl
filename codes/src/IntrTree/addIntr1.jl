function addIntr1!(Root::AbstractTreeNode,
    A::AbstractTensorMap,site::Int64,name::String,fermionic::Bool,strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    # tempOpr = LocalOperator(A,name,site,strength)
    # addIntr1!(Root,tempOpr,Z)
    Leave = InteractionTreeLeave(A,site,name,fermionic,strength,Z)
    addIntr1!(Root,Leave)
end

function addIntr1!(Root::InteractionTreeNode,Leave::InteractionTreeLeave)
    Opr = LocalOperator(Leave)
    Z = Leave.Z
    current_node = Root
    current_site = 1

    # add the identity
    while current_site < Opr.site

        if isnothing(Z)
            tempIdOpr = IdentityOperator(getIdTensor(Opr),current_site)
        else
            tempIdOpr = LocalOperator(Z,"Z",current_site,1)
        end

        indId = findfirst(x -> isequal(x.Opr,tempIdOpr),current_node.children)
        if isnothing(indId)
            addchild!(current_node,tempIdOpr)
            current_node = current_node.children[end]
        else
            current_node = current_node.children[indId]
        end

        current_site += 1
    end

    addchild!(current_node, Opr)
    current_node.children[end].Leave = Leave
    typeof(Root) <: ObservableTreeNode && (current_node.children[end].name = Tuple(Opr.site))
end

# function addIntr1!(Root::AbstractTreeNode,
#     Opr::LocalOperator,
#     Z::Union{Nothing,AbstractTensorMap})
#     current_node = Root
#     current_site = 1

#     # add the identity
#     while current_site < Opr.site

#         if isnothing(Z)
#             tempIdOpr = IdentityOperator(getIdTensor(Opr),current_site)
#         else
#             tempIdOpr = LocalOperator(Z,"Z",current_site,1)
#         end

#         indId = findfirst(x -> isequal(x.Opr,tempIdOpr),current_node.children)
#         if isnothing(indId)
#             addchild!(current_node,tempIdOpr)
#             current_node = current_node.children[end]
#         else
#             current_node = current_node.children[indId]
#         end

#         current_site += 1
#     end

#     # add the onsite Opr 
#     addchild!(current_node, Opr)
#     # current_node.children[end].Opr.strength = strength
#     typeof(Root) <: ObservableTreeNode && (current_node.children[end].name = Tuple(Opr.site))
# end