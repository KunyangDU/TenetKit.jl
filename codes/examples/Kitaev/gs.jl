using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Kitaev/data"

D = 2^7
Lx = 4
Ly = 4
params = (Jx=0.0,Jy=1.0,Jz = 0.0)

Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

H = TrivialHamiltonian(Latt;params...,ϵ=0)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 7)
showQuantSweep(lsEg)

# @save "$(dataname)/sweep/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @save "$(dataname)/sweep/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
# @save "$(dataname)/sweep/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
# xbonds,ybonds,zbonds = getxyzbonds(Latt)


