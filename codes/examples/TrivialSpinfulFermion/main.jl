using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
# some problems left (up and down's anticommutation)

Lx = 8
Ly = 1

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinfulFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)

t = 1
U = 0
μ = 0
H = let 
    Root = InteractionTreeNode()
    LocalSpace = TrivialSpinfulFermion

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.n,i,"n",-μ,nothing)
        addIntr!(Root,LocalSpace.nd,i,"nd",U,nothing)
    end
    
    for pair in neighbor(Latt)
        addIntr!(Root,LocalSpace.F₊⁺F₊,pair,("F₊⁺","F₊"),-t,LocalSpace.Z)
        addIntr!(Root,LocalSpace.F₊F₊⁺,pair,("F₊","F₊⁺"),t,LocalSpace.Z)
        addIntr!(Root,LocalSpace.F₋⁺F₋,pair,("F₋⁺","F₋"),-t,LocalSpace.Z)
        addIntr!(Root,LocalSpace.F₋F₋⁺,pair,("F₋","F₋⁺"),t,LocalSpace.Z)
    end

    AutomataSparseMPO(InteractionTree(Root),size(Latt))
end
D = 2^8

ψ,lsE = DMRG2!(ψ,H,D;LanczosLevel = 30)
showQuantSweep(lsE)

@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinfulFermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

#=     for k in -π:π/4:π
        addObs!(Obs.forest, (LocalSpace.F₊⁺F₊,LocalSpace.F₊F₊⁺,LocalSpace.n₊), Latt, [k,0], (("F₊ₖ⁺","F₊ₖ"),("F₊ₖ","F₊ₖ⁺"),"n₊"),LocalSpace.Z)
        addObs!(Obs.forest, (LocalSpace.F₋⁺F₋,LocalSpace.F₋F₋⁺,LocalSpace.n₋), Latt, [k,0], (("F₋ₖ⁺","F₋ₖ"),("F₋ₖ","F₋ₖ⁺"),"n₋"),LocalSpace.Z)
    end =#

    calObs!(Obs,ψ)
end
@show sum([Obs.values["n"][(i,)] for i in 1:size(Latt)])
Obs.values
