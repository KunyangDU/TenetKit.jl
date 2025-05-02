using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/U1"

D = 2^9
Lx = 4
Ly = 4
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
lsF = []
lsE = []

@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN!(lsβ[1], H, ρ;D=2^7 )
Env = Environment([ρ,H,ρ'])
initialize!(Env)
Z = norm(ρ)^2
push!(lsF,-log(Z)/lsβ[1]/2)
push!(lsE,real(scalar(Env)) / Z)

lsρ,lsinfo,lsFt,lsEt = tanTRG1!(Env, lsβ;trunc = truncdim(D) & truncbelow(1e-12))
lsF = vcat(lsF,lsFt)
lsE = vcat(lsE,lsEt)

@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE


