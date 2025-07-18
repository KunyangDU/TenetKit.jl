using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Kitaev_XCHC/data"

D = 400
DSETTN = 50
Lx = 6
Ly = 4
Latt = XCHC(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (Jx=1,Jy=1,Jz=1)

H = Hamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-15:1:-1), 1:10)

# @save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

# SETTN1!(lsβ[1], H, ρ;
# trunc = truncdim(DSETTN) & truncbelow(1e-8),tol = 1e-8,
# algo = CBEalgo(dynamicSVD(1.2,4),NoStruc(),0,DSETTN),max_order = 10)
SETTN1!(lsβ[1], H, ρ;trunc = truncdim(2^5))

Z = normalize!(ρ) ^ 2 
lsρ,lsinfo,lsF,lsE = tanTRG2!(ρ,H, lsβ;lnZ = log(Z),trunc = truncdim(D) & truncbelow(1e-8))


# @save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
# @save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
# @save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE


