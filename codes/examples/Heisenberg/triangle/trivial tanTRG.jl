using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/triangle/trivial"

D = 2^9
Lx = 6
Ly = 4
Latt = XCTria(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (J=1,)
Latt = YCSqua(Lx,Ly)
h = 1
pinh = PINVEC120(Latt,h)
H = TrivialHamiltonian(Latt; params...,pinh=pinh)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-20:1:-1), 1:10)
lsF = []
lsE = []

@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN!(lsβ[1], H, ρ;D=2^7)
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


