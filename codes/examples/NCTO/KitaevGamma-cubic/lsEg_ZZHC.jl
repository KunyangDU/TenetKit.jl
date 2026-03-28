using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/NCTO/KitaevGamma-cubic/data/ZZHC"

D = 64
Lx = 2
Ly = 3

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)

Latt = ZZHoneyComb(Lx,Ly)
# @save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

# Hf = 0.

θ = 0.0
ϕ = 0.0

K = 1
# Γ = 0.0
for Γ′ in 0.01
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params1_Kitaev = (K = K, Γ′ = Γ′)

# for Hf in 0.24:0.04:0.36
Hf = 0.0

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

println("$(Lx)x$(Ly), D = $(D), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/lsinfo_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsinfo
lsEg = [info.E[end] for info in lsinfo]
@save "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg

GC.gc()
end

# map(x -> showdomain(x.A),ψ.ts)