using TensorKit
include("../../src/TNKit.jl")
include("model.jl")
dataname = "examples/Hubbard/data"

Lx = 6
Ly = 1
Latt = iYCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 100
params = (U = 8,μ=4)
H = Hamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([Rep[U₁×SU₂]((0, 0) => 1),], size(Latt)+1)
    ρ = IdDenseMPO(U₁SU₂Fermion.PhySpace, AuxSpaces)
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


