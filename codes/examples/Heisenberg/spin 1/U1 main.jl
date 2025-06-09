using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
dataname = "examples/Heisenberg/spin 1/data/U1"

D = 3^5
for Lx in [60,80,100]
# Lx = 8
Ly = 1
params = (Jz=1, Jxy = 1/2, )

Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -2*size(Latt):2*size(Latt) ),], Lx*Ly-1))
    randMPS(U₁Spin1.PhySpace ,AuxSpace)
end

H = U1Hamiltonian(Latt;params...)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg ./ size(Latt) .- 1/4)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
end
