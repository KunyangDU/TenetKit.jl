using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")
#= 
Fermion complexity
=#
dataname = "examples/TrivialSpinfulFermion/data"

D = 2^7
Lx = 4
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
for μ in -0.2:-0.2:-1
params = (t = 1, μ = μ, U = 0)
Latt = YCSqua(Lx,Ly)

H = Hamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinfulFermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-15:1:-1), 1:10)
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN1!(lsβ[1], H, ρ;trunc = truncdim(2^6))
Z = normalize!(ρ) ^ 2 
lsρ,lsinfo,lsF,lsE = tanTRG1!(ρ,H, lsβ;lnZ = log(Z),trunc = truncdim(D) & truncbelow(1e-8))

@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE



end