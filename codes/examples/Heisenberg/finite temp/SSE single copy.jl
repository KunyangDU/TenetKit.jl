using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^6
# for Ly in 4:2:20,Lx in Ly:2:20 
Lx = 64
Ly = 1

Latt = YCSqua(Lx,Ly)
for Hz in vcat(0:0.2:0.8,1.2:0.2:3)
params = (J=1.0, Δ = 1.0, Hz = Hz)
Ops = Vector{InteractionTreeNode}(undef,size(Latt))

H = TrivialHamiltonian(Latt;params...,returnnode = true)

@time lsroot = let
    lsroot = CompositeObservableTreeNode[]
    
    node_replace!(x,obj) = let 
        x.A = x.A*obj.A - obj.A*x.A 
        x.name = "[$(x.name),$(obj.name)]"
    end

    for i in 1:size(Latt)
        rootup = InteractionTreeNode()
        addIntr!(rootup,TrivialSpinOneHalf.S₊,i,"S₊",false,1,nothing)
        S₋ = LocalOperator(TrivialSpinOneHalf.S₋,"S₋",i,1)
        rootdown = commutate(H,S₋)
        tmpr = CompositeObservableTreeNode((rootup,rootdown))
        buildtree!(tmpr)
        push!(lsroot,cutparent!(tmpr))
    end
    lsroot
end

map(lsroot[2:end]) do root 
    merge!!(lsroot[1],(root))
end
Obs = Observable(lsroot[1])

# @load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
# @load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

# for (iβ,β) in enumerate(lsβ)
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
lsI = Vector{Dict}(undef,length(lsρ))
for (i,ρ) in enumerate(lsρ)
    calObs!(Obs,ρ;destroy = false,
    cachesize = 2*(get_num_threads_julia() - 1))
    lsI[i] = Obs.values
    GC.gc()
end

@save "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
end
