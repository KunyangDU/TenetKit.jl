using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Heisenberg/spin 1/data/trivial"


D = 3^4
for Lx in [4,6,8,10,20,40,60,80,100]
# Lx = 8
Ly = 1
params = (J=1,)

Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOne.PhySpace ,AuxSpace)
end

H = TrivialHamiltonian(Latt;params...)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg ./ size(Latt) .- 1/4)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
end

