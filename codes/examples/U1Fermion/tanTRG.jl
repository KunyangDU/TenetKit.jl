using TensorKit
include("../../src/iMPS.jl")
include("model.jl")




#= 
Fermion complexity
=#

Lx = 6
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 30
params = (μ=0,)
Ndop = 0
H = Hamiltonian(Latt;params...)
ρ = let 
    AuxSpaces = repeat([Rep[U₁](0 => 1),],size(Latt)+1)
    ρ = IdDenseMPO(U₁Fermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-10:1:-1), 1:10)

SETTN!(lsβ[1], H, ρ;D=20)
lsρ,lsinfo = tanTRG1!(ρ, H, lsβ, D)

@save "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ

