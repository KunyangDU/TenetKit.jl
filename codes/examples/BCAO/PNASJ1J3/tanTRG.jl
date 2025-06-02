using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#
dataname = "examples/BCAO/PNASJ1J3/data"


D = 2^4
Lx = 2
Ly = 2
Ds = 2^4

params = (Hx = 0., J1xy = -1.0, J1z = -0.158, D = 0.0132, E = -0.0132, J3xy = 0.329, J3z = -0.112)

println("$(Lx)x$(Ly), D = $(D), params = $(params)")
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

H = TrivialHamiltonian(Latt; params...)

ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-20:1:-1), 1:10)

@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
SETTN1!(lsβ[1], H, ρ;
trunc = truncdim(Ds),tol = 1e-6,
algo = CBEalgo(dynamicSVD(1.2,2),NoStruc(),0,Ds,-Inf),max_order = 10)
# SETTN2!(lsβ[1], H, ρ;
# trunc = truncdim(Ds),tol = 1e-6,max_order = 10)
Z = normalize!(ρ)^2

lsρ,lsinfo,lsF,lsE = tanTRG1!(ρ, H, lsβ;lnZ = log(Z), trunc = truncdim(D) & truncbelow(1e-12))


@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE


