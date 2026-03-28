using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/KitaevGamma-cubic/data/OHHC"

D = 128
L = 3

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)

Latt = OHHoneyComb(L)
@save "$(dataname)/Latt_$(L).jld2" Latt


θ = 0.0
ϕ = pi / 2

K = 1
Γ = -0.8

params1_Kitaev = (K = K, Γ = Γ)

# for Hf in 0:0.04:0.8
Hf = 0.0

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

println("size = $(size(Latt)), D = $(D), \nparams_Kitaev = $(params_Kitaev)")

H = TrivialHamiltonian(Latt;params_Kitaev...,
shift = [0,0], direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])

ψ = let 
    AuxSpace = repeat([ℂ^1,], size(Latt))
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end
# @load "$(dataname)/ψ_$(L)_$(D)_$(params_Kitaev).jld2" ψ

lsEg,lsinfo = DMRG2!(ψ, H;trunc = truncdim(D) & truncbelow(1e-16),N = 20)
showQuantSweep(lsEg)

@save "$(dataname)/lsEg_$(L)_$(D)_$(params_Kitaev).jld2" lsEg
@save "$(dataname)/lsinfo_$(L)_$(D)_$(params_Kitaev).jld2" lsinfo
@save "$(dataname)/ψ_$(L)_$(D)_$(params_Kitaev).jld2" ψ
# end





