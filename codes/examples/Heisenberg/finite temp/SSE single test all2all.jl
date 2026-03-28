using TensorKit,Serialization

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^6
# for Ly in 4:2:20,Lx in Ly:2:20 
Lx = 20
Ly = 1

Latt = YCSqua(Lx,Ly)
# for Hz in vcat(0:0.2:0.8,1.2:0.2:3)
params = (J=1.0, Δ = 1.0, Hz = 1.0)
Ops = Vector{InteractionTreeNode}(undef,size(Latt))

H = TrivialHamiltonian(Latt;params...,returnnode = true)

@time root = let
    
    node_replace!(x,obj) = let 
        x.A = x.A*obj.A - obj.A*x.A 
        x.name = "[$(x.name),$(obj.name)]"
    end
    rootup = InteractionTreeNode()
    rootdown = InteractionTreeNode()
    for i in 1:size(Latt)
        addIntr!(rootup,TrivialSpinOneHalf.S₊,i,"S₊",false,1,nothing)
        S₋ = LocalOperator(TrivialSpinOneHalf.S₋,"S₋",i,1)
        rootcom = commutate(H,S₋)
        merge!!(rootdown,rootcom)
    end
    root = CompositeObservableTreeNode((rootup,rootdown))
    buildtree!(root)
end

Obs = Observable(root)
treesize(root)

# for (iβ,β) in enumerate(lsβ)
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
ρ = lsρ[end]
calObs!(Obs,ρ;destroy = false,isdisk = true,showtimes = 100)



# @save "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
# end
