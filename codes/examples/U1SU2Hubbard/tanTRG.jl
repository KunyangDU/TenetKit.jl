using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#
foldername = "examples/U1SU2Hubbard/data"
Lx = 10
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(foldername)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 300
params = (U = 0,)
H = Hamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([Rep[U₁×SU₂]((0, 0) => 1),], size(Latt)+1)
    ρ = IdDenseMPO(U₁SU₂Fermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-10:1:-1), 1:10)

SETTN!(lsβ[1], H, ρ;D=D)
lsρ,lsinfo = tanTRG1!(ρ, H, lsβ, D)

@save "$(foldername)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "$(foldername)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ


