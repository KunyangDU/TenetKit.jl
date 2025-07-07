using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/trivial"

D = 2^6
Lx = 10
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (J=1,)
Latt = YCSqua(Lx,Ly)

H = TrivialHamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-20:1:-1), 1:10)

@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN1!(lsβ[1], H, ρ;trunc = truncdim(2^4))
Z = normalize!(ρ) ^ 2 
lsρ,lsinfo,lsF,lsE = tanTRG2!(ρ,H, lsβ;lnZ = log(Z),trunc = truncdim(D) & truncbelow(1e-12))


@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE


