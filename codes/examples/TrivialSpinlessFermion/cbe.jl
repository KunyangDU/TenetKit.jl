using TensorKit,CairoMakie
include("../../src/iMPS.jl")
include("model.jl")

Lx = 50
Ly = 1
D = 150

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end
Latt = YCSqua(Lx,Ly)
μ = 0
H = Hamiltonian(Latt;μ=μ)

lsE = DMRG1!(ψ,H,D,1e-3;Nsweep = 5,return_error = false,cbe = true)
showQuantSweep(lsE .- sum(@. -2cos(pi*(1:div(Lx*Ly,2))/(Lx*Ly+1))))

