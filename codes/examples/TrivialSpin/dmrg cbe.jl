using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 10
Ly = 1

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)

hx = 0
H = Hamiltonian(Latt)
D = 20

lsE = DMRG1!(ψ,H,D,1e-12;cbe=true,Nsweep = 5)
showQuantSweep(lsE ./ size(Latt) .- 1/4)

#= @time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinOneHalf

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",nothing)
    end

    for i in 1:size(Latt),j in i+1:size(Latt)
        addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),nothing)
    end

    calObs!(Obs,ψ)
end

Obs.values =#
