using TensorKit,JLD2,KrylovKit
include("../../src/TenetKit.jl")
include("model.jl")


Lx = 6
Ly = 1
D = 2^6
Latt = YCSqua(Lx,Ly)
@save "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,],Lx*Ly)
    randMPS(TrivialSpinlessFermion.PhySpace ,AuxSpace)
end

μ = 0
t = 1
H = Hamiltonian(Latt;μ=μ)
lsE,lsinfo = DMRG1!(ψ,H;trunc = truncdim(D) & truncbelow(1e-12))
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))


@save "examples/TrivialSpinlessFermion/data/ψ_$(Lx)x$(Ly)_$(D).jld2" ψ




