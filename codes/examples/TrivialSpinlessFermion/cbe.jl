using TensorKit,CairoMakie
include("../../src/iMPS.jl")
include("model.jl")

Lx = 3
Ly = 3
D = 100

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
μ = 0
H = Hamiltonian(Latt;μ=μ)

lsE = DMRG1!(ψ,H,D,1e-8;Nsweep = 5)
showQuantSweep(lsE .- ue(100,Lx,Ly)*Lx*Ly)


