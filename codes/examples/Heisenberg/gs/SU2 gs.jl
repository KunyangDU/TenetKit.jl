
using TensorKit
include("../../../src/TNKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/SU2"

    # @tensor x[-1,-2;-3] ≔ El.A[] * obj.A[] * h.A[] * Er.A[]
D = 2^9
# lsLx = 6:2:12
# for Lx in lsLx
Lx = 8
Ly = 6
params = (J=1,)
Latt = YCSqua(Lx,Ly)

ψ = let 
    AuxSpace = vcat(Rep[SU₂](0 => 1),repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], size(Latt)-1))
    randMPS(SU₂Spin.PhySpace ,AuxSpace)
end

@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

H = SU2Hamiltonian(Latt;params...)

lsEg,lsinfo = DMRG!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
# end

lsEg


