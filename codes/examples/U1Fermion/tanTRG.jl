using TensorKit
include("../../src/iMPS.jl")
include("model.jl")


#= 
Fermion complexity
=#

Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "examples/U1Fermion/data/Latt_$(Lx)x$(Ly).jld2" Latt

D = 128
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
lsF = []
lsE = []
@save "examples/U1Fermion/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN!(lsβ[1], H, ρ;D=32)
Env = Environment([ρ,H,ρ'])
initialize!(Env)
Z = norm(ρ)
push!(lsF,-log(Z)/lsβ[1])
push!(lsE,scalar(Env) / Z^2)

lsρ,lsinfo,lsFt,lsEt = tanTRG1!(Env, lsβ;trunc = truncdim(D) & truncbelow(1e-12))
lsF = vcat(lsF,lsFt)
lsE = vcat(lsE,lsEt)

@save "examples/U1Fermion/data/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "examples/U1Fermion/data/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "examples/U1Fermion/data/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE

