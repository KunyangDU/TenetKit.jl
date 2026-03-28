using TensorKit

include("../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/YuanYao/data/spin1"
PBC = true
D = 243
Ly = 1
params = (J₁ = 1.0, J₂ = 0.0)

# Lx = 20
for Lx in 120
@show Lx

Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOne.PhySpace ,AuxSpace)
end

H = MGHamiltonian1(Latt;params...,PBC = PBC)

lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12), N =40, Stol = 1e-6)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params)_$(PBC).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params)_$(PBC).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params)_$(PBC).jld2" ψ
# # end

lsEg

end