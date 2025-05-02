using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/U1"

D = 2^7
params = (Jz = 1,Jxy = 0.5,h=0)

lsLx = 4:2:12
for Lx in lsLx
    Ly = 4
    Latt = YCSqua(Lx,Ly)
    @save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt


    ψ = let 
        AuxSpace = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt)//2 :1//2:size(Latt)//2 ),], Lx*Ly-1))
        randMPS(U₁Spin.PhySpace ,AuxSpace)
    end

    H = U1Hamiltonian(Latt;params...)
    lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
    showQuantSweep(lsEg / size(Latt) .- 1/4)
    @save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
    @save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
    @save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
end





