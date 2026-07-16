using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/triangular"
IS_DISK[] = true
diskdir!()

D = 256
Lx = 6
Ly = 6
params = (J = 1.0 , Hx = 0.0, Hy = 5.0, Hz = 0.0)

Latt = XCTria(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

H = TrivialHamiltonian(Latt;params...)

lsEg,lsinfo = DMRG1!(ψ, H; trunc = truncdim(D), N = 20)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
