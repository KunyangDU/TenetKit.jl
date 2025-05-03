using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/SU2/connect"
# for D in [2^7,2^8]
D = 2^7
Lx = 14
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (J=1,)

H = SU2Hamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([Rep[SU₂]((0) => 1),], size(Latt)+1)
    ρ = IdDenseMPO(SU₂Spin.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-20:1:-1),1)

@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN!(lsβ[1], H, ρ;D=2^7 )
Z = normalize!(ρ)^2

lsρ,lsinfo,lsF,lsE = tanTRG1!(ρ, H, lsβ;lnZ = log(Z),trunc = truncdim(D) & truncbelow(1e-12))

@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
# end

