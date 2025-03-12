using TensorKit,JLD2,KrylovKit
include("../../src/iMPS.jl")
include("model.jl")

Lx = 4
Ly = 4
D = 120

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
μ = 0
H = Hamiltonian(Latt;μ=μ)
lsE,lsinfo = DMRG1!(ψ,H;trunc = truncdim(D) & truncbelow(1e-6))
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))
#A = ψ.ts[1]
#zerovector(ψ.ts[1],Float64)
#@save "examples/TrivialSpinlessFermion/"
