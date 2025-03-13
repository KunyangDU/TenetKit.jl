using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#
foldername = "examples/U1SU2Hubbard/data"
Lx = 8
Ly = 1
Latt = iYCSqua(Lx,Ly)
L = size(Latt)
@save "$(foldername)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 200
params = (U = 0,)
H = Hamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([Rep[U₁×SU₂]((0, 0) => 1),], size(Latt)+1)
    ρ = IdDenseMPO(U₁SU₂Fermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-15:1:-1), 1:10)
lsF = []
lsE = []

SETTN!(lsβ[1], H, ρ;D=32 )
Env = Environment([ρ,H,ρ'])
initialize!(Env)
push!(lsF,-log(norm(ρ))/lsβ[1])
push!(lsE,scalar(Env))

lsρ,lsinfo,lsFt,lsEt = tanTRG1!(Env, lsβ;trunc = truncdim(D) & truncbelow(1e-12))
lsF = vcat(lsF,lsFt)
lsE = vcat(lsE,lsEt)
@save "$(foldername)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "$(foldername)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(foldername)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(foldername)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE


