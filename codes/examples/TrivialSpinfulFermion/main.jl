using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
# some problems left (up and down's anticommutation)

Lx = 6
Ly = 1

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinfulFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)

params = (t = 1,U = 0,μ = 0)
H = Hamiltonian(Latt;params...)
D = 100

lsE = DMRG2!(ψ,H,D)
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))

#= @time "calculate observables" begin
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
Obs.values =#
