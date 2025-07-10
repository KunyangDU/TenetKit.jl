using TensorKit
include("../../../src/iMPS.jl")
include("model.jl")
dataname = "examples/BCAO/arxiv2025/data"

D = 2^7
Lx = 4
Ly = 4
# for J1 in -0.45:-0.05:-0.55
# params1_Kitaev = (J1 = -0.59, K1 = -1, Γ1 = 0.53, Γ1′ = 0.11)
# params23 = (J2 = -0.038, J3xy = 0.31, J3z = 0.0092)
# paramsh = (pinh=1.,)

params1_Kitaev = (J1 = -0.59, K1 = 0.0, Γ1 = 0., Γ1′ = 0.0)
params23 = (J2 = -0.038, J3xy = 0.32, J3z = 0.0)
paramsh = (pinh=0.,)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params23,paramsh)
params_Kitaev = merge(params1_Kitaev,params23,paramsh)

println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end
# pinh = map(x -> x*1,repeat([[0.,1.,0.],],4Ly))
pinh = params.pinh .*vcat(repeat([[0.,1.,0.],],2Ly),repeat([[0.,-1.,0.],],2Ly))
H = TrivialHamiltonian(Latt;params...,pinh = pinh)

lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 5)
showQuantSweep(lsEg)

@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ
# end


