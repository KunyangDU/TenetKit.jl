using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Kitaev_XCHC/data"


D = 2^8
# lsLx = 4:2:12
# for Lx in lsLx
Lx = 6
Ly = 4
params = (Jx=1,Jy=1,Jz=1)

Latt = XCHC(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

H = Hamiltonian(Latt;params...)

lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 6)
# showQuantSweep(lsEg ./ size(Latt) .- 1/4)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
# end

lsEg



