using TensorKit,CairoMakie
include("../../src/iMPS.jl")
include("model.jl")


Lx = 8
Ly = 1

N = Lx*Ly

Latt = YCSqua(Lx,Ly)

params = (μ = 0,)
H = Hamiltonian(Latt;params...)
Nop = ParticleNumber(Latt)
D = 2^8

ρ = let 
    AuxSpaces = repeat([ℂ^1,], Lx*Ly+1)
    ρ = IdDenseMPO(TrivialSpinlessFermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-5:1:-1),1:10)
ρ = SETTN!(lsβ[1],H,deepcopy(ρ);D = D)
lsρ = tanTRG2!(ρ, H, lsβ, D;LanczosLevel = 15)
#lsρ = SETTN!(lsβ,H,ρ;max_order = 30,F_tol = 1e-16)
@save "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_tanTRG.jld2" lsβ
@save "examples/TrivialSpinlessFermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params)_tanTRG.jld2" lsρ




