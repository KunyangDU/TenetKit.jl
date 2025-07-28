
function addIntr3!(
    Root::AbstractTreeNode,
    As::NTuple{3,AbstractTensorMap},
    sites::NTuple{3,Int64},
    names::NTuple{3,String},
    fermionic::NTuple{3,Bool},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap}
    )
    Leave = InteractionTreeLeave(As,sites,names,fermionic,strength,Z)
    addIntr3!(Root,Leave)
end

function addIntr3!(Root::AbstractTreeNode,Leave::InteractionTreeLeave{3})
    OprL = LocalOperator(Leave.A[1],Leave.name[1],Leave.site[1])
    OprC = LocalOperator(Leave.A[2],Leave.name[2],Leave.site[2])
    OprR = LocalOperator(Leave.A[3],Leave.name[3],Leave.site[3],Leave.strength)

    Z = Leave.Z
    isZ = isfermionic(Leave.fermionic)

    @assert OprL.site < OprC.site < OprR.site

    current_node = Root
    current_site = 1

    # add the identity
    while current_site < OprR.site

        tempOpr = isZ ? LocalOperator(Z,"Z",current_site) : IdentityOperator(getIdTensor(OprL),current_site)
        if current_site == OprL.site
            tempOpr = OprL
            isZ = isZ ⊻ Leave.fermionic[1]
        end

        if current_site == OprC.site
            tempOpr = OprC
            isZ = isZ ⊻ Leave.fermionic[2]
        end

        indId = findfirst(x -> isequal(x.A,tempOpr),current_node.children)
        if isnothing(indId)
            addchild!(current_node,tempOpr)
            current_node = current_node.children[end]
        else
            current_node = current_node.children[indId]
        end

        current_site += 1
    end

    # add the right A
    if !isnothing(Z)
        _addZ!(OprR,Z)
    end

    addchild!(current_node, OprR)
    
    if typeof(Root) <: ObservableTreeNode
        # current_node.children[end].name = tuple(OprL.site,OprR.site)
        current_node.children[end].Leave = ObservableTreeLeave(Leave)
    else
        current_node.children[end].Leave = Leave
    end
end



