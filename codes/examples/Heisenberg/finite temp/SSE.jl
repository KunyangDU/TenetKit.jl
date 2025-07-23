using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

function traceback(A::InteractionTreeNode)
    let node = deepcopy(A)
        while true
            if isnothing(node.parent) | node.parent.Opr.site == 0
                node.parent = nothing
                return node
            else
                x = deepcopy(node.parent)
                x.children = [node,]
                node = x
            end
        end
    end
end

function TimerOutputs.merge!(A::AbstractTreeNode, B::AbstractTreeNode, value::Function = nodevalue)
    ind = findfirst(x -> isequal(value(x), value(B)), A.children)
    if !isnothing(ind)
        for c in B.children
            merge!(A.children[ind], c)
        end
    else
        B.parent = nothing
        addchild!(A,B)
        return A
    end
end

function relevent_node(root::InteractionTreeNode, site::Int64, dismiss::Function = (x -> false))
    A = []
    for p in PreOrderDFS(root)
        !(typeof(p.Opr) <: LocalOperator) && continue
        p.Opr.site ≠ site && continue
        !dismiss(p.Opr) && push!(A,traceback(p))
    end

    # list = [root,]
    # while !isempty(list)
    #     l = popat!(list,1)
    #     if l.Opr.site < site 
    #         push!(list,l.children...)
    #     else
    #         typeof(l.Opr) <: LocalOperator && !dismiss(l.Opr) && push!(A,traceback(l))
    #     end
    # end
    
    relroot = InteractionTreeNode()

    for a in A
        merge!(relroot,a)
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
    root = relevent_node(O, obj.site, x -> norm(x.A*obj.A - obj.A*x.A) ≈ 0)
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
    name::Union{Nothing, AbstractVector, Tuple}

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
    CompositeTreeNode() = CompositeTreeNode((nothing,nothing))
end
AbstractTrees.nodevalue(node::CompositeTreeNode) = node.A

trivial(x::T) where T<: AbstractTreeNode = T(trivial(nodevalue(x)))
function trivial_next(x::InteractionTreeNode)
    y = trivial(nodevalue(x))
    y.site += 1
    return InteractionTreeNode(y)
end

function buildtree!(A::CompositeTreeNode)
    if !reduce(&, map(x ->  isempty(x.children), A.A))
        for a in Iterators.product(map(x -> isempty(x.children) ? [trivial(x),] : x.children, A.A)...)
            x = CompositeTreeNode(a)
            x.name = deepcopy(A.name)
            for (i,aa) in enumerate(a)
                if !isnothing(aa.Opr.name)
                    push!(x.name[i], aa.Opr.site)
                end
            end
            
            addchild!(A, x)
        end
    end

    A.A = map(x -> isnothing(x) ? nothing : nodevalue(x), A.A)
    A.name = Tuple(Tuple.(A.name))

    for c in A.children
        buildtree!(c)
    end

    return A
end

function pushright(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor x[-1;-2] ≔ ht.A[2,6] * objt.A[1,3,-2,2] * hb.A[5,1] * objb.A[-1,6,5,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function pushright(::IdentityOperator{1}, objt::DenseMPOTensor{4}, hb::LocalOperator{1, 1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor x[-1;-2] ≔ objt.A[1,3,-2,2] * hb.A[5,1] * objb.A[-1,2,5,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function pushright(::IdentityOperator{1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor x[-1;-2] ≔ objt.A[1,3,-2,2] * objb.A[-1,2,1,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

function pushright(ht::LocalOperator{1, 1}, objt::DenseMPOTensor{4}, ::IdentityOperator{1}, objb::AdjointMPOTensor{4}, EnvL::LeftEnvironmentTensor{2})
    @tensor x[-1;-2] ≔ ht.A[2,5] * objt.A[1,3,-2,2] * objb.A[-1,5,1,4] * EnvL.A[4,3]
    return LeftEnvironmentTensor(x)
end

D = 2^5
Lx = 4
Ly = 1

Latt = YCSqua(Lx,Ly)
J = 1
# for H in [1,2]
H = 1
params = (J=J, H = H)

H =  let Root = InteractionTreeNode(), LocalSpace=TrivialSpinOneHalf
    
    for pair in neighbor(Latt)
        addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋),pair,("S₊","S₋"),J/2,nothing)
        addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊),pair,("S₋","S₊"),J/2,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",-H,nothing)
    end

    InteractionTree(Root)
end
H



# root = CompositeTreeNode((nothing,nothing))
# root.name = ((),())
# for i in 1:size(Latt)
#     S₋ = LocalOperator(TrivialSpinOneHalf.S₋,"S₋",i,1)
#     # @show H.Root.children[1]
#     rootc = commutate(H.Root.children[1],S₋)
#     rootcomm = InteractionTree(rootc)
#     rootup = let Root = InteractionTreeNode()
#         addIntr!(Root,TrivialSpinOneHalf.S₊,i,"S₊",1,nothing)
#         InteractionTree(Root)
#     end
#     x = CompositeTreeNode((rootup.Root,rootcomm.Root))
#     x.name =  [[],[]]
#     buildtree!(x)
#     merge!(root,x.children[1])
# end
# root = cutparent!(root.children[1])

# to

@time root = let
    rootup = InteractionTreeNode()
    rootdown = InteractionTreeNode()
    node_replace!(x,obj) = let 
        x.A = x.A*obj.A - obj.A*x.A 
        x.name = "[$(x.name),$(obj.name)]"
    end

    for i in 1:size(Latt)
        S₋ = LocalOperator(TrivialSpinOneHalf.S₋,"S₋",i,1)
        addIntr!(rootup,TrivialSpinOneHalf.S₊,i,"S₊",1,nothing)
        rootc = commutate(H.Root.children[1],S₋)
        merge!(rootdown,rootc)
    end
        
    root = CompositeTreeNode((rootup,cutparent!(rootdown.children[1])))
    root.name =  [[],[]]
    buildtree!(root)
end
Base.summarysize(root)/1024/1024


@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ

ρ = lsρ[end]
EnvL = LeftEnvironmentTensor(isometry(ℂ^1,ℂ^1))
root.Env = EnvL
for p in PreOrderDFS(root)
    site = maximum(x -> x.site, p.A)
    if site != 0
        p.Env = (isnan(p.A[1].strength) ? 1 : p.A[1].strength) * (isnan(p.A[2].strength) ? 1 : p.A[2].strength) * pushright(p.A[1],ρ.ts[site],p.A[2],ρ.ts[site]',p.Env)
    end

    if isempty(p.children)
        p.value = real(_scalar(p.Env))
        p.Env = nothing
    else
        for r in p.children
            r.Env = p.Env
        end
    end

    p.Env = nothing
end

data = Dict()
for l in Leaves(root)
    data[l.name] = l.value + get(data, l.name, 0)
end
data


# data



    # lsI[iρ] = data
# end

# @save "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
# end
# data

# collect(Leaves(root))

# root


# cpt.A[1].children[1]


# cpt4 = CompositeTreeNode((cs4.Root,cr4.Root))
# cpt3 = CompositeTreeNode((cs3.Root,cr3.Root))


# buildtree!(cpt3)
# buildtree!(cpt4)

# merge!(cpt3,cpt4.children[1])

# cpt3
# cpt4
# cpt3

# treeheight(cpt)

# A = cpt.children[end].children[end].children[end].children[end].children[end].children[end].children[end].children[end].children[end]

# map(x -> !isnothing(x),A.A)
# collect(Iterators.product(map(x -> isnothing(x) | isempty(x.children) ? [nothing,] : x.children, A.A)...))[2]
# A.A[1].children


# cpt

# (Sx,Sx) == (Sx,nothing)

