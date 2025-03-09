using TensorKit,JLD2,KrylovKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1
D = 10

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
μ = 0
H = Hamiltonian(Latt;μ=μ)
lsE = DMRG1!(ψ,H,truncdim(D) & truncbelow(1e-6);Nsweep = 3)
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))
#A = ψ.ts[1]
#zerovector(ψ.ts[1],Float64)
#@save "examples/TrivialSpinlessFermion/"
