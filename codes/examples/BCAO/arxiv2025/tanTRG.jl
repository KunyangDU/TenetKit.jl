using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")

#= 
Fermion complexity
=#
dataname = "examples/BCAO/arxiv2025/data"

D = 2^6
Lx = 2
Ly = 2
Ds = 2^4

params1_Kitaev = (J1 = -0.59, K1 = -1, Γ1 = 0.53, Γ1′ = 0.11)
params23 = (J2 = -0.038, J3xy = 0.31, J3z = 0.0092, Hx = 0.)
paramsh = (pinh=0.,)

Latt = ZZHoneyComb(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params23,paramsh)
params_Kitaev = merge(params1_Kitaev,params23,paramsh)
println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

pinh = params.pinh .*vcat(repeat([[0.,1.,0.],],2Ly),repeat([[0.,-1.,0.],],2Ly))
H = TrivialHamiltonian(Latt; params...,pinh=pinh)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat(2. .^ (-20:1:-1), 1:10)

@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsβ
SETTN1!(lsβ[1], H, ρ;
trunc = truncdim(Ds) & truncbelow(1e-8),tol = 1e-8,
algo = CBEalgo(dynamicSVD(1.2,4),NoStruc(),0,Ds,1e-8),max_order = 10)
Z = normalize!(ρ)^2

lsρ,lsinfo,lsF,lsE = tanTRG1!(ρ, H, lsβ;lnZ = log(Z), trunc = truncdim(D) & truncbelow(1e-12))


@save "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsρ
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsE


