using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

ObservableTreeNode
D = 2^7
# for Ly in 4:2:20,Lx in Ly:2:20 
Lx = 14
Ly = 1

Latt = YCSqua(Lx,Ly)
J = 1
# for H in [1,2]
H = 1
params = (J=J, )

H =  let Root = InteractionTreeNode(), LocalSpace=TrivialSpinOneHalf
    
    
    for pair in neighbor(Latt)
        addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋),pair,("S₊","S₋"),(false,false),J/2,nothing)
        addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊),pair,("S₋","S₊"),(false,false),J/2,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),(false,false),J,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",false,-H,nothing)
    end

    Root
end
H

# relevent_node(H.Root, 2)


# root = CompositeObservableTreeNode((nothing,nothing))
# root.name = ((),())
# @time for i in 2
#     S₋ = LocalOperator(TrivialSpinOneHalf.S₋,"S₋",i,1)
#     # @show H.Root.children[1]
#     rootc = commutate(H.Root.children[1],S₋)
#     rootcomm = InteractionTree(rootc)
#     rootup = let Root = InteractionTreeNode()
#         addIntr!(Root,TrivialSpinOneHalf.S₊,i,"S₊",false,1,nothing)
#         InteractionTree(Root)
#     end
#     x = CompositeObservableTreeNode((rootup.Root,rootcomm.Root))
#     x.name =  [[],[]]
#     buildtree!(x)
#     merge!(root,x.children[1])
# end
# root = cutparent!(root.children[1])
# @show root
# 1
# memo = Base.summarysize(root)/1024/1024

# end


@time root = let
    rootup = InteractionTreeNode()
    rootdown = InteractionTreeNode()
    node_replace!(x,obj) = let 
        x.A = x.A*obj.A - obj.A*x.A 
        x.name = "[$(x.name),$(obj.name)]"
    end

    for i in 1:size(Latt)
        addIntr!(rootup,TrivialSpinOneHalf.S₊,i,"S₊",false,1,nothing)
        S₋ = LocalOperator(TrivialSpinOneHalf.S₋,"S₋",i,1)
        rootc = commutate(H,S₋)
        # @show rootc
        merge!(rootdown,rootc)
    end
        
    root = CompositeObservableTreeNode((rootup,cutparent!(rootdown.children[1])))
    # root.name =  [[],[]]
    buildtree!(root)
end
# Base.summarysize(root)/1024/1024
# InteractionTreeNode
treesize(root)

@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ

ρ = lsρ[end]
Obs = Observable(root)
calObs!(Obs,ρ)
# EnvL = LeftEnvironmentTensor(isometry(ℂ^1,ℂ^1))
# root.Env = EnvL
# for p in PreOrderDFS(root)
#     site = maximum(x -> x.site, p.A)
#     if site != 0
#         p.Env = (isnan(p.A[1].strength) ? 1 : p.A[1].strength) * (isnan(p.A[2].strength) ? 1 : p.A[2].strength) * pushright(p.A[1],ρ.ts[site],p.A[2],ρ.ts[site]',p.Env)
#     end

#     if isempty(p.children)
#         p.Leave.value = real(_scalar(p.Env))
#         p.Env = nothing
#     else
#         for r in p.children
#             r.Env = p.Env
#         end
#     end

#     p.Env = nothing
# end

# data = Dict()
# for l in Leaves(root)
#     data[(l.Leave.name,l.Leave.site)] = l.Leave.value
# end
# data

# for k in keys(data)
#     k[2][1][1] ≠ 2 && continue 
#     @show k,data[k]
# end


# data



    # lsI[iρ] = data
# end

# @save "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
# end
# data

# collect(Leaves(root))

# root


# cpt.A[1].children[1]


# cpt4 = CompositeObservableTreeNode((cs4.Root,cr4.Root))
# cpt3 = CompositeObservableTreeNode((cs3.Root,cr3.Root))


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

