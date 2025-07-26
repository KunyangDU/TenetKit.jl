using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")

dataname = "examples/TrivialSpinlessFermion/data"
Lx = 6
Ly = 1
D = 2^7

N = Lx*Ly

Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
H = Hamiltonian(Latt;params...)

ρ = let 
    AuxSpaces = repeat([ℂ^1,], Lx*Ly+1)
    ρ = IdDenseMPO(TrivialSpinlessFermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-20:1:-1),1:10)
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

ρ = SETTN1!(lsβ[1],H,deepcopy(ρ);trunc = truncdim(2^6))
Z = normalize!(ρ)^2

lsρ,lsinfo,lsF,lsE = tanTRG1!(ρ, H, lsβ;lnZ = log(Z),trunc = truncdim(D) & truncbelow(1e-12))

@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
