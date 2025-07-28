
function addIntr4!(
    Root::AbstractTreeNode,
    As::NTuple{4,AbstractTensorMap},
    sites::NTuple{4,Int64},
    names::NTuple{4,String},
    fermionic::NTuple{4,Bool},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap}
    )
    Leave = InteractionTreeLeave(As,sites,names,fermionic,strength,Z)
    addIntr4!(Root,Leave)
end

function addIntr4!(Root::AbstractTreeNode,Leave::InteractionTreeLeave{4})
    OprL = LocalOperator(Leave.A[1],Leave.name[1],Leave.site[1])
    OprCL = LocalOperator(Leave.A[2],Leave.name[2],Leave.site[2])
    OprCR = LocalOperator(Leave.A[3],Leave.name[3],Leave.site[3])
    OprR = LocalOperator(Leave.A[4],Leave.name[4],Leave.site[4],Leave.strength)

    Z = Leave.Z
    isZ = isfermionic(Leave.fermionic)

    @assert OprL.site < OprCL.site < OprCR.site < OprR.site

    current_node = Root
    current_site = 1

    # add the identity
    while current_site < OprR.site

        tempOpr = isZ ? LocalOperator(Z,"Z",current_site) : IdentityOperator(getIdTensor(OprL),current_site)
        if current_site == OprL.site
            tempOpr = OprL
            isZ = isZ ⊻ Leave.fermionic[1]
        end

        if current_site == OprCL.site
            tempOpr = OprCL
            isZ = isZ ⊻ Leave.fermionic[2]
        end

        if current_site == OprCR.site
            tempOpr = OprCR
            isZ = isZ ⊻ Leave.fermionic[3]
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



