using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1

Latt = YCSqua(Lx,Ly)
Ndop = 2

ψ = let
    AuxSpace = vcat(Rep[U₁](Ndop // 2 => 1), repeat([Rep[U₁](i => 1 for i in -(abs(Ndop) + 1):1//2:(abs(Ndop)+1)),], size(Latt) - 1))
    randMPS(U₁Fermion.PhySpace, AuxSpace)
end

μ = 0
H = Hamiltonian(Latt;μ=μ)

D = 2^4

lsE = DMRG2!(ψ,H,D,1e-10)
showQuantSweep(lsE)

@time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = U₁Fermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

    calObs!(Obs,ψ)
end

@show sum([Obs.values["n"][(i,)] for i in 1:size(Latt)])
Obs.values

