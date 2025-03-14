using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#
foldername = "examples/U1SU2Hubbard/data"
Lx = 6
Ly = 1
Latt = iYCSqua(Lx,Ly)
L = size(Latt)
@save "$(foldername)/Latt_$(Lx)x$(Ly).jld2" Latt

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
lsF = []
lsE = []

@save "$(foldername)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN!(lsβ[1], H, ρ;D=16 )
Env = Environment([ρ,H,ρ'])
initialize!(Env)
Z = norm(ρ)^2
push!(lsF,-log(Z)/lsβ[1]/2)
push!(lsE,scalar(Env) / Z)

lsρ,lsinfo,lsFt,lsEt = tanTRG1!(Env, lsβ;trunc = truncdim(D) & truncbelow(1e-12))
lsF = vcat(lsF,lsFt)
lsE = vcat(lsE,lsEt)
@save "$(foldername)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(foldername)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(foldername)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE


