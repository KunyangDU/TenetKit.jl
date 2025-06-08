
using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/SU2"

D = 2^6
# lsLx = 6:2:12
# for Lx in lsLx
Lx = 4
Ly = 4
params = (J=1,)
Latt = YCSqua(Lx,Ly)

ψ = let 
    AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], size(Latt)-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end

@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

H = SU2Hamiltonian(Latt;params...)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg ./ size(Latt) .- 1/4)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
# end



