using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1

Latt = YCSqua(Lx,Ly)
Ndop = 0

ψ = let
    AuxSpace = vcat(Rep[U₁×U₁]((Ndop, 0) => 1), repeat([Rep[U₁×U₁]((i, j) => 1 for i in -(abs(Ndop) + 1):(abs(Ndop)+1) for j in -1:1//2:1),], size(Latt) - 1))
    randMPS(U₁U₁Fermion.PhySpace, AuxSpace)
end

params = (U = 0,)

H = Hamiltonian(Latt;params...)

D = 200

lsE = DMRG1!(ψ,H,D)
showQuantSweep(lsE)

#= 
@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = U₁U₁Fermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

    calObs!(Obs,ψ)
end

@show sum([Obs.values["n"][(i,)] for i in 1:size(Latt)])
Obs.values =#

