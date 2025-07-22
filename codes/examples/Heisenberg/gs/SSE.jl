using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

function traceback(A::InteractionTreeNode)
    let node = A
        while true
            if isnothing(node.parent) | node.parent.Opr.site == 0
                node.parent = nothing
                return node
            end
            x = deepcopy(node.parent)
            x.children = [node,]
            node = x
        end
    end
end

function relevent_node(root::AbstractTreeNode, site::Int64, dismiss::Function = (x -> false))
    A = []
    for p in PreOrderDFS(root)
        if typeof(p.Opr) <: LocalOperator && p.Opr.site == site
            !dismiss(p.Opr.A) && push!(A,traceback(p))
        end
    end
    
    relroot = InteractionTreeNode()

    for a in A
        current_obj_node = a
        current_root_node = relroot
        while true
            ind = findfirst(x -> isequal(x.Opr, current_obj_node.Opr), current_root_node.children)
            if !isnothing(ind)
                current_root_node = current_root_node.children[ind]
                current_obj_node = current_obj_node.children[1]
            else
                current_obj_node.parent = nothing
                addchild!(current_root_node,current_obj_node)
                break
            end
        end
    end

    return relroot
end

function _onsite_replace!(root::AbstractTreeNode, site::Int64, node_replace!::Function)
    for p in PreOrderDFS(root)
        if typeof(p.Opr) <: LocalOperator && p.Opr.site == site
            node_replace!(p.Opr)
        end
    end

    return root
end


function commutate(O::AbstractTreeNode, obj::LocalOperator)
    root = relevent_node(O, obj.site, x -> norm(x*obj.A - obj.A*x) ≈ 0)
    node_replace!(x) = let 
        x.A = x.A*obj.A - obj.A*x.A 
        x.name = "[$(x.name),$(obj.name)]"
    end
    return _onsite_replace!(root, obj.site, node_replace!)
end

mutable struct CompositeTreeNode{N} <: AbstractTreeNode where {N}
    A::NTuple{N,Any}
    parent::Union{Nothing,CompositeTreeNode}
    children::Vector{CompositeTreeNode}
    Env::Union{Nothing, AbstractEnvironmentTensor}
    value::Union{Nothing, Number}
    name::Union{Nothing, Tuple}

    function CompositeTreeNode(
        A::NTuple{N,Any},
        parent::CompositeTreeNode,
        children::Vector{CompositeTreeNode}=CompositeTreeNode[],
    ) where N
        return new{N}(A,parent,children,nothing,NaN,nothing)
    end

    function CompositeTreeNode(
        A::NTuple{N,Any},
        children::Vector{CompositeTreeNode}=CompositeTreeNode[],
    ) where N
        return new{N}(A,nothing,children,nothing,NaN,nothing)
    end
    # CompositeTreeNode() = CompositeTreeNode(IdentityOperator(0))
end
AbstractTrees.nodevalue(node::CompositeTreeNode) = node.A


D = 2^7
# lsLx = 4:2:12
# for Lx in lsLx
Lx = 4
Ly = 4
params = (J=1,)

Latt = YCSqua(Lx,Ly)
# @save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

# ψ = let 
#     AuxSpace = repeat([ℂ^1,], Lx*Ly)
#     randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
# end
J = 2
H = 0

H =  let Root = InteractionTreeNode(), LocalSpace=TrivialSpinOneHalf
    
    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.SxSx,pair,("Sx","Sx"),J,nothing)
        addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sy"),J,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J,nothing)
        # addIntr!(Root,LocalSpace.SySy,pair,("Sy","Sz"),J,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",-H,nothing)
    end

    # return AutomataSparseMPO(InteractionTree(Root),size(Latt))  
    InteractionTree(Root)
end


Sx = LocalOperator(TrivialSpinOneHalf.Sx,"Sx",4,1)

cr = InteractionTree(commutate(H.Root,Sx))

cs = let Root = InteractionTreeNode(), LocalSpace=TrivialSpinOneHalf
    
    addIntr!(Root,LocalSpace.Sx,4,"Sx",1,nothing)

    InteractionTree(Root)
end



cpt = CompositeTreeNode((cs.Root,cr.Root))

# cpt.A[1].children[1]

function buildtree!(A::CompositeTreeNode)
    #  && return A

    if reduce(|, map(x -> !isnothing(x),A.A)) && !reduce(&, map(x -> isnothing(x) ? true : isempty(x.children), A.A))
        for a in Iterators.product(map(x -> !(!isnothing(x) && !isempty(x.children)) ? [nothing,] : x.children, A.A)...)
            addchild!(A, CompositeTreeNode(a))
        end
    else
        return A
    end

    for c in A.children
        buildtree!(c)
    end

    return A
end



buildtree!(cpt)

treeheight(cpt)

# A = cpt.children[end].children[end].children[end].children[end].children[end].children[end].children[end].children[end].children[end]

# map(x -> !isnothing(x),A.A)
# collect(Iterators.product(map(x -> isnothing(x) | isempty(x.children) ? [nothing,] : x.children, A.A)...))[2]
# A.A[1].children





