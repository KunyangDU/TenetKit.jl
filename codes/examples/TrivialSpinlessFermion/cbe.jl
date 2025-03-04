using TensorKit,JLD2,KrylovKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1
D = 400

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
μ = 0
H = Hamiltonian(Latt;μ=μ)
lsE = DMRG2!(ψ,H,D,1e-8;Nsweep = 5)
#showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))
#A = ψ.ts[1]
#zerovector(ψ.ts[1],Float64)
#@save "examples/TrivialSpinlessFermion/"
