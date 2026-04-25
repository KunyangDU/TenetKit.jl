using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/Kitaev/data"

D = 2^6
Lx = 1
Ly = 2
params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.01)
Hx,Hy,Hz = params.Ha * [1,-1,0] / sqrt(2) + params.Hb * [1,1,-2] / sqrt(6) + params.Hc * [1,1,1] / sqrt(3)

Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

H = TrivialHamiltonian(Latt;params...,Hx = Hx,Hy = Hy,Hz = Hz)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 20)
showQuantSweep(lsEg)

# @save "$(dataname)/sweep/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @save "$(dataname)/sweep/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
# @save "$(dataname)/sweep/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
# xbonds,ybonds,zbonds = getxyzbonds(Latt)


