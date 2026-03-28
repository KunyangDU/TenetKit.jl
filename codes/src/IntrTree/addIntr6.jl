function addIntr6!(
    Root::AbstractTreeNode,
    As::NTuple{6,AbstractTensorMap},
    sites::NTuple{6,Int64},
    names::NTuple{6,String},
    fermionic::NTuple{6,Bool},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap}
    )
    Leave = InteractionTreeLeave(As,sites,names,fermionic,strength,Z)
    addIntr6!(Root,Leave)
end

function addIntr6!(Root::AbstractTreeNode,Leave::InteractionTreeLeave{6})
    Oprs = Vector{LocalOperator}(undef,6)
    Oprs = map(i -> LocalOperator(Leave.A[i],Leave.name[i],Leave.site[i]), 1:6)
    Oprs[end].strength = Leave.strength

    Z = Leave.Z
    isZ = isfermionic(Leave.fermionic)
    sites = map(x -> x.site,Oprs)
    fermionics = collect(Leave.fermionic)
    @assert issorted(sites) "not sorted"
    right_site = pop!(sites)

    current_node = Root
    current_site = 1

    # add the identity
    while current_site < right_site

        tempOpr = isZ ? LocalOperator(Z,"Z",current_site) : IdentityOperator(getIdTensor(Oprs[1]),current_site)

        if !isempty(sites) && current_site == sites[1]
            tempOpr = popat!(Oprs,1)
            isZ = isZ ⊻ popat!(fermionics,1)
            popat!(sites,1)
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
        _addZ!(Oprs[1],Z)
    end

    addchild!(current_node, Oprs[1])
    
    if typeof(Root) <: ObservableTreeNode
        current_node.children[end].Leave = ObservableTreeLeave(Leave)
    else
        current_node.children[end].Leave = Leave
    end
end