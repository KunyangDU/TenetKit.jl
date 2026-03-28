using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")

# λ = 0.5
totalname = "examples/J1J2chain/plateau/data"
Ly = 1
Lx = 60
D = 128

Latt = YCSqua(Lx,Ly)
@save "$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

for H in 0:0.04:0.4
params = (J1 = -1, J2 = 1.0, J1xy = 0.0, Hx = 0.0, Hy = 0.0, Hz = H)

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

H = Hamiltonian(Latt;params...)
lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 20)
showQuantSweep(lsEg)

@save "$(totalname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(totalname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
end

