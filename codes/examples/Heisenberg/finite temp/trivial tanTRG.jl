using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/trivial"

D = 256
Lx = 2
Ly = 4
Ds = 32
Latt = YCRect(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
for Hz in [1.0,]
params = (J = 1.0, Δ = 1.0, Hz = Hz)

H = TrivialHamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat((1.0 + 0.5) .^ (-20:1:-1), 1:0.5:10)
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN1!(lsβ[1], H, ρ;trunc = truncdim(Ds))
Z = normalize!(ρ) ^ 2 
lsρ,lsinfo,lsF,lsE = tanTRG1!(ρ,H, lsβ;lnZ = log(Z),trunc = truncdim(D) & truncbelow(1e-8))

@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE

end
# lsE

