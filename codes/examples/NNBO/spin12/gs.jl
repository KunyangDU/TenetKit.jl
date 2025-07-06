using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
include("../geometry.jl")
dataname = "examples/NNBO/spin12/data"

D = 2^7
Lx = 4
Ly = 4
params = (J=1,)

Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

params1_1_Kitaev = (J1 = -1, K1 = 0.6, Γ1 = 0, Γ1′ = 0)
params1_3_JDH = (J3 = 1, D = 3)
paramsh = (h=0.0, H = 0.)

params1_cry = _Cub2Cry(params1_1_Kitaev)
params = merge(params1_cry,params1_3_JDH,paramsh)
params_Kitaev = merge(params1_1_Kitaev,params1_3_JDH,paramsh)

H = TrivialHamiltonian(Latt;params...)

lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
# showQuantSweep(lsEg ./ size(Latt) .- 1/4)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ



