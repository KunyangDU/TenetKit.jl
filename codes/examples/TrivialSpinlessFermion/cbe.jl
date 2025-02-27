using TensorKit,CairoMakie
include("../../src/iMPS.jl")
include("model.jl")

Lx = 8
Ly = 1
D = 100

ψ = let 
    AuxSpace = vcat([ℂ^min(i,D) for i in 2 .^ (0:div(Lx*Ly,2)-1)],[ℂ^min(i,D) for i in 2 .^ (div(Lx*Ly,2):-1:1)])
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)
μ = 0
H = Hamiltonian(Latt;μ=μ)

lsE = DMRG1!(ψ,H,D,1e-12;Nsweep = 5,return_error = false,cbe = false)
showQuantSweep(lsE .- sum(@. -2cos(pi*(1:div(Lx*Ly,2))/(Lx*Ly+1))))

