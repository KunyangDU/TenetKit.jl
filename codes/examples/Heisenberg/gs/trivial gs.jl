using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"
IS_DISK[] = true
diskdir!()

D = 100
Lx = 4
Ly = 4
params = (J=1, Δ = 1)

Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace, isdisk = true)
end

H = TrivialHamiltonian(Latt;params...)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12), Etol = 1e-20, N = 10,isdisk = true)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

