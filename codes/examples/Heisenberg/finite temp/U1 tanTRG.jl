using TensorKit
include("../../../src/TNKit.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/U1"

D = 2^7
Lx = 14
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (Jz = 1,Jxy = 0.5,h=0)

H = U1Hamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([Rep[U₁]((0) => 1),], size(Latt)+1)
    ρ = IdDenseMPO(U₁Spin.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-20:1:-1), 1:10)

@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN1!(lsβ[1], H, ρ; trunc = truncdim(2^5))
Z = normalize!(ρ)^2

lsρ,lsinfo,lsF,lsE = tanTRG2!(ρ,H, lsβ;lnZ = log(Z),trunc = truncdim(D) & truncbelow(1e-12))

@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE


