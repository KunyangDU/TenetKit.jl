using TensorKit
include("../../src/iMPS.jl")
include("model.jl")
dataname = "examples/NNBO/data/H"

D = 3^4
Lx = 4
Ly = 4
for H in 0:0.4:2.8
params1_Kitaev = (J1 = -1, K1 = 0.6, Γ1 = 0, Γ1′ = 0)
params3DH = (J3 = 1, D = -3,  H = H)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params3DH)
params_Kitaev = merge(params1_Kitaev,params3DH)

println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOne.PhySpace ,AuxSpace)
end
H = TrivialHamiltonian(Latt;params...)

lsEg,lsinfo = DMRG1!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg)

@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ
end

