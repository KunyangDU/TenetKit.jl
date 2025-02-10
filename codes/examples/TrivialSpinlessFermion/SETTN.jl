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
lsβ = vcat(2. .^ (-15:1:-1),1)
lsρ = SETTN!(lsβ,H,ρ;max_order = 20,F_tol = 1e-8,D=D)
@save "examples/TrivialSpinlessFermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_SETTN.jld2" lsβ
@save "examples/TrivialSpinlessFermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params)_SETTN.jld2" lsρ




