using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma/data/PCHC"
figurename = "NCTO/KitaevGamma/figures/PCHC"
tailname = ""

D = 2^8
Lx = 4
Ly = 4
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)



θ = pi / 2
ϕ = 0.0



lsHf = 0:0.04:0.8


K = 1
Γ  = 0.0
params1_Kitaev = (J1 = J, K1 = K, Γ1 = Γ, Γ1′ = Γ1′)
params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

lsE = zeros(length(lsHf))

for (i,Hf) in enumerate(lsHf)



Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

params_H = (Hx = Hx, Hy = Hy, Hz = Hz)

params = merge(params1,params23,params_H)
params_Kitaev = merge(params1_Kitaev,params23,params_H)

# println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
# @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
lsE[i] = lsEg[end]
end


lsχ = [-(lsE[i] + lsE[i+2] - 2lsE[i+1])/(lsHf[i+2] - lsHf[i+1])/(lsHf[i+1] - lsHf[i]) for i in 1:length(lsE)-2]





fig = Figure()
ax = Axis(fig[1,1])
scatterlines!(ax,lsHf[1:end-2],lsχ)

display(fig)
# lsχ

lsE