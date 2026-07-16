using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/U1"

D = 512
params = (Jz = 1.0,Jxy = 0.5,Hz = 2.0,J′ = 1.0)

Lx = 6
Ly = 6
Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = vcat(Rep[U₁](0 => 1),repeat([Rep[U₁](i => 1 for i in -1:1//2:1 ),], size(Latt)-1))
    randMPS(U₁Spin.PhySpace ,AuxSpace)
end

H = U1Hamiltonian(Latt;params...)
lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 10)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

lsEg






