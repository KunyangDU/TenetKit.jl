using TensorKit,JLD2
include("../../src/iMPS.jl")
include("model.jl")

Lx = 6
Ly = 1
D = 600

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
μ = 0
H = Hamiltonian(Latt;μ=μ)
lsE = DMRG1!(ψ,H,D,1e-8;Nsweep = 5,λ=1.2)

showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))

#@save "examples/TrivialSpinlessFermion/"

