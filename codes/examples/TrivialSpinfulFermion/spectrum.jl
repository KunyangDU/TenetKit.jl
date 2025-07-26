using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")
#= 
Fermion complexity
=#
dataname = "examples/TrivialSpinfulFermion/data"
D = 2^7
Lx = 4
Ly = 1
Latt = YCSqua(Lx,Ly)
# @load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
for μ in -0.2:-0.2:-1
params = (t = 1, μ = μ, U = 0)

@time roots = let LocalSpace = TrivialSpinfulFermion
    
    # rootup = InteractionTreeNode()
    rootdown = InteractionTreeNode()
    

    for i in 1:size(Latt)
        # addIntr!(rootup,LocalSpace.F₊,i,"F₊",true,1,LocalSpace.Z)
        addIntr!(rootdown,LocalSpace.F₊⁺,i,"F₊⁺",true,1,LocalSpace.Z)
    end

    root = Vector(undef,size(Latt))
    for i in 1:size(Latt)
        rootup = InteractionTreeNode()
        addIntr!(rootup,LocalSpace.F₊,i,"F₊",true,1,LocalSpace.Z)
        root[i] = CompositeObservableTreeNode((rootup,rootdown))
        buildtree!(root[i])
    end

    # root = CompositeObservableTreeNode((rootup,rootdown))
    # buildtree!(root)
    root
end
# treesize(root)



Obs2 = Observable()
for i in 1:size(Latt)
    addObs!(Obs2,TrivialSpinfulFermion.n,i,"n",false,nothing)
end

@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ

ρ = lsρ[end]

data = Dict()

calObs!(Obs2, ρ;destroy = false)
data["obs"] = Obs2.values

for (i,root) in enumerate(roots)
    Obs1 = Observable(root)
    calObs!(Obs1,ρ;destroy = false)
    data["G_$(i)"] = Obs1.values
end

# Obs1 = Observable(roots)
# calObs!(Obs1,ρ;destroy = false)
# data["G"] = Obs1.values

@save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data

end
