using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
include("../geometry.jl")
dataname = "examples/NNBO/data"

D = 3^4
Lx = 4
Ly = 4
# for J1 in -0.45:-0.05:-0.55
params1_Kitaev = (J1 = -1, K1 = 0.6, Γ1 = 0, Γ1′ = 0)
params3DH = (J3 = 1, D = -3,  H = 0.)
paramsh = (h=0.01,)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params3DH,paramsh)
params_Kitaev = merge(params1_Kitaev,params3DH,paramsh)

println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOne.PhySpace ,AuxSpace)
end

pinh = map(x -> x * paramsh.h, vcat(repeat([[0.,0.,1.],],2Ly), repeat([[0.,0.,-1.],],2Ly)))
H = TrivialHamiltonian(Latt;params...,pinh = pinh)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg)

@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ


