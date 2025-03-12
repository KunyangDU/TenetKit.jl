using TensorKit
include("../../src/iMPS.jl")
include("../model.jl")

#= 
Fermion complexity
=#
foldername = "data"
Lx = 8
Ly = 4
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(foldername)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2^10
Dsettn = 2^8
params = (U = 0,)
lsβ = vcat(2. .^ (-10:1:-1), 1:10)

H = Hamiltonian(Latt; params...)

ρ = let 
    AuxSpaces = repeat([Rep[U₁×SU₂]((0, 0) => 1),], size(Latt)+1)
    ρ = IdDenseMPO(U₁SU₂Fermion.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end


@save "$(foldername)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
lsρ = []
lsinfo = []
SETTN!(lsβ[1], H, ρ;D=Dsettn)
push!(lsρ,deepcopy(ρ))
@save "$(foldername)/ρ_$(Lx)x$(Ly)_$(D)_$(params)_$(1)_$(lsβ[1]).jld2" ρ

for i in 2:length(lsβ)
    Env = Environment([ρ,H,ρ'])
    Alg = TDVPalgo(DoubleSite(),NoAlgorithm(),truncdim(D) & truncbelow(1e-6),0,1e-4,TDVPDefaultLanczos)
    info = TDVPinfo()
    TDVP!(Env::Environment{3,L}, Alg::TDVPalgo, info::TDVPinfo)
    ρ = Env.layer[1]
    @save "$(foldername)/ρ_$(Lx)x$(Ly)_$(D)_$(params)_$(i)_$(lsβ[i]).jld2" ρ
    @save "$(foldername)/info_$(Lx)x$(Ly)_$(D)_$(params)_$(i)_$(lsβ[i]).jld2" info
    push!(lsρ,deepcopy(ρ))
    push!(lsinfo,deepcopy(info))
end

# lsρ,lsinfo = tanTRG2!(ρ, H, lsβ;tol = 100, trunc = truncdim(D) & truncbelow(1e-6))

@save "$(foldername)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ


