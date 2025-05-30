using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/triangle/pin"

D = 100

Lx = 4
Ly = 4

Latt = XCTria(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

h = 0
h = PINVEC120(Latt,h)

params = (J=1,H = 1.)
ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

H = TrivialHamiltonian(Latt;params...,pinh=h)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg ./ size(Latt) .- 1/4)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ


