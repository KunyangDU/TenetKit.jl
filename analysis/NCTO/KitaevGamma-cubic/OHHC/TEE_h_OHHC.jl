using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/OHHC"
figurename = "NCTO/KitaevGamma-cubic/figures/OHHC"
tailname = ""

D = 256
L = 3

@load "$(dataname)/Latt_$(L).jld2" Latt


θ = 0.0
ϕ = pi / 2

K = 1
Γ = 0.0

params1_Kitaev = (K = K, Γ = Γ)

# Hf = 0.0
lsHf = 0:0.04:0.8
lsTEE = zeros(length(lsHf))

flux_Latt = _OHTria(L-1,((1.0, 0.0),(1/2, sqrt(3)/2));scale = sqrt(3))
flux_center_sites = map(x -> collect(coordinate(flux_Latt,x)), 1:size(flux_Latt))
@show flux_center_sites
for (i,Hf) in enumerate(lsHf)

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)
# Hcx,Hcy,Hcz = Hx * [1,-1,0] + Hy * [1,1,-2] + Hz * [1,1,1]

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

# println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/lsEg_$(L)_$(D)_$(params_Kitaev).jld2" lsEg
@load "$(dataname)/gsdata_$(L)_$(D)_$(params_Kitaev).jld2" gsdata

lsTEE[i] = gsdata["TEE"]["TEE"]
end

figsize = (height = 200,width = 400)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "size = $(size(Latt)), D = $(D), K = $(K), Γ = $(Γ)\n θ = $(θ/pi)π, ϕ = $(ϕ/pi)π",
xlabel = "h",ylabel = L"\gamma_E\ /\ \ln 2")

scatterlines!(ax,lsHf,lsTEE)

resize_to_layout!(fig)
display(fig)

save("$(figurename)/flux_h_$(L)_$(D)_$(params_Kitaev).png",fig)
save("$(figurename)/flux_h_$(L)_$(D)_$(params_Kitaev).pdf",fig)


