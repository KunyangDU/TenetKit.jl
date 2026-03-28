using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/KitaevGamma-cubic/data/OHHC"

D = 6
L= 1

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)

@load "$(dataname)/Latt_$(L).jld2" Latt

# Hf = 0.

θ = 0.0
ϕ = pi / 2

K = 1
Γ = 0.0

params1_Kitaev = (K = K, Γ = Γ)

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

println("$(size(Latt))), D = $(D), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/lsEg_$(L)_$(D)_$(params_Kitaev).jld2" lsEg
@load "$(dataname)/lsinfo_$(L)_$(D)_$(params_Kitaev).jld2" lsinfo
@load "$(dataname)/ψ_$(L)_$(D)_$(params_Kitaev).jld2" ψ


@tensor S1[-1,-2;-3,-4] ≔ ψ.ts[1].A[-1,1,-3] * ψ.ts[1].A'[-2,-4,1]
@tensor S3[-1,-2;-3,-4] ≔ ψ.ts[3].A[-1,1,-3] * ψ.ts[3].A'[-2,-4,1]
@tensor S5[-1,-2;-3,-4] ≔ ψ.ts[5].A[-1,1,-3] * ψ.ts[5].A'[-2,-4,1]

@tensor Stot[-1,-2;-3,-4] ≔ S1[1,2,3,1] * ψ.ts[2].A[3,-1,-3] * ψ.ts[2].A'[-2,2,-4]
@tensor Stot[-1,-2;-3,-4] ≔ Stot[-1,1,2,-4] * S3[2,-2,-3,1]
@tensor Stot[-1,-2,-3;-4,-5,-6] ≔ Stot[-1,1,2,-5] * ψ.ts[4].A[2,-2,-4] * ψ.ts[4].A'[-3,1,-6]
@tensor Stot[-1,-2,-3;-4,-5,-6] ≔ Stot[-1,-2,1,2,-5,-6] * S5[2,-3,-4,1]
@tensor Stot[-1,-2,-3;-4,-5,-6] ≔ Stot[-1,-2,1,2,-4,-5] * ψ.ts[6].A[2,-3,3] * ψ.ts[6].A'[3,1,-6]

f = eigen(Stot.data)
S = f.values
sum(- S .* log.(S))


