using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/triangle"

D = 2^7

Lx = 6
Ly = 6
Latt = XCTria(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

lsH = 0:0.2:5
for H in lsH
    @show H
    params = (Jz = 1,Jxy = 0.5,H=H)

    ψ = let 
        AuxSpace = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -size(Latt):1//2:size(Latt) ),], Lx*Ly-1))
        randMPS(U₁Spin.PhySpace ,AuxSpace)
    end

    H = U1Hamiltonian(Latt;params...)
    lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
    showQuantSweep(lsEg / size(Latt) .- 1/4)
    @save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
    @save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
    @save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
end





