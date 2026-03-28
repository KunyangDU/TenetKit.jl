using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/Kitaev/data"

D = 20
Lx = 2
Ly = 2

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)

Latt = PCHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@show getxyzbonds(Latt;shift = [1/2,sqrt(3)/2], direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])
θ = 0.0
K = -1.0
Γ = 0.0

params1_Kitaev = (J1 = J, K1 = K, Γ1 = Γ, Γ1′ = Γ1′)
params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end
inputHxy = round(0.0*sqrt(3);digits = 3)
Hx = 0.
Hy = 0.
Hz = inputHxy

params_H = (Hx = Hx, Hy = Hy, Hz = Hz)

params = merge(params1,params23,params_H)
params_Kitaev = merge(params1_Kitaev,params23,params_H)

println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

H = TrivialHamiltonian(Latt;params...,
shift = [1/2, sqrt(3)/2],
direction = [[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-12),N = 10)
showQuantSweep(lsEg)

@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@save "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsinfo
@save "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" ψ






