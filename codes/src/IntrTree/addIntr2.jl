function addIntr2!(
    Root::AbstractTreeNode,
    As::NTuple{2,AbstractTensorMap},
    sites::NTuple{2,Int64},
    names::NTuple{2,String},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap}
    )
    # OprL,OprR = map(x -> LocalOperator(As[x],names[x],sites[x],strength),1:2)
    # if sites[1] > sites[2]
    #     A,B = _swap(As)
    #     OprL = LocalOperator(A,names[2],sites[2])
    #     OprR = LocalOperator(B,names[1],sites[1],strength)
    # else
    #     OprL = LocalOperator(As[1],names[1],sites[1])
    #     OprR = LocalOperator(As[2],names[2],sites[2],strength)
    # end
    OprL = LocalOperator(As[1],names[1],sites[1])
    OprR = LocalOperator(As[2],names[2],sites[2],strength)

    addIntr2!(Root,OprL,OprR,Z)
end

function addIntr2!(
    Root::AbstractTreeNode,
    OprL::LocalOperator,OprR::LocalOperator,
    Z::Union{Nothing,AbstractTensorMap}
    )
    # @assert OprL.site < OprR.site
    if OprL.site > OprR.site
        OprL.A,OprR.A = _swap(OprL.A,OprR.A,Z)
        OprL.site,OprR.site = OprR.site,OprL.site
        OprL.name,OprR.name = OprR.name,OprL.name
    end

    current_node = Root
    current_site = 1

    # add the identity
    while current_site < OprR.site

        tempOpr = let 
            if current_site < OprL.site
                IdentityOperator(getIdTensor(OprL),current_site)
            elseif current_site == OprL.site
                OprL
            elseif isnothing(Z)
                IdentityOperator(getIdTensor(OprL),current_site)
            else
                LocalOperator(Z,"Z",current_site)
            end
        end

        indId = findfirst(x -> isequal(x.Opr,tempOpr),current_node.children)
        if isnothing(indId)
            addchild!(current_node,tempOpr)
            current_node = current_node.children[end]
        else
            current_node = current_node.children[indId]
        end

        current_site += 1
    end

    # add the right Opr
    if !isnothing(Z)
        _addZ!(OprR,Z)
    end

    addchild!(current_node, OprR)
    typeof(Root) <: ObservableTreeNode && (current_node.children[end].name = tuple(OprL.site,OprR.site))
end

function _addZ!(OprR::LocalOperator, Z::AbstractTensorMap)
    OprR.A = _addZ(OprR.A,Z)
    OprR.name = string("Z",OprR.name)
end

function _addZ(A::AbstractTensorMap{S₁,2,1}, Z::AbstractTensorMap{S₂,1,1}) where {S₁,S₂}
    @tensor tmp[-1 -2;-3] ≔ Z[-1,1] * A[1,-2,-3]
    return tmp
end

function _addZ(A::AbstractTensorMap{S₁,1,1}, Z::AbstractTensorMap{S₂,1,1}) where {S₁,S₂}
    @tensor tmp[-1;-2] ≔ Z[-1,1] * A[1,-2]
    return tmp
end

# TODO: Z ≠ nothing
function _swap(A::AbstractTensorMap,B::AbstractTensorMap,::Nothing,tol::Float64 = 1e-12)
    @tensor AB′[-1,-2;-3,-4] ≔ A[-1,1,-4] * B[-2,1,-3]
    A′,Λ,B′ = tsvd(AB′,(2,3),(1,4),trunc = truncbelow(tol))
    return permute(A′,(1,),(3,2)),permute(Λ*B′,(2,1),(3,))
end
_swap(AB::NTuple{2,AbstractTensorMap},::Nothing,tol::Float64 = 1e-12) = _swap(AB...,tol)

