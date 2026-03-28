using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/OHHC"
figurename = "NCTO/KitaevGamma-cubic/figures/OHHC/flux"
tailname = ""

D = 256
L = 3

@load "$(dataname)/Latt_$(L).jld2" Latt



for Hf in 0:0.04:0.8

θ = pi / 2
ϕ = pi / 2

K = 1
Γ = 0.0

params1_Kitaev = (K = K, Γ = Γ)

# Hf = 0.0


# for (i,Hf) in enumerate(lsHf)


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

obsdata = gsdata["obs"]

flux = Dict()
fluxsites = []
fluxvalues = []
flux_key = filter(x -> length(x[1]) == 6, collect(keys(obsdata)))
for kdc in flux_key, k in keys(obsdata[kdc])
    push!(fluxsites,sum(collect.(map(x -> coordinate(Latt,x),k[1]))) / 6)
    push!(fluxvalues,obsdata[kdc][k])
end


figsize = (height = 300,width = 300)

fluxcolors = get(colorschemes[:seismic],fluxvalues * 2^6,(-1,1))
fig = Figure()
ax = Axis(fig[1,1],autolimitaspect = true;figsize...,
title = "size = $(size(Latt)), D = $(D), K = $(K), Γ = $(Γ)\n $((Hx,Hy,Hz))")

# @show fluxcolors
polyHexagon!(ax,fluxsites,fluxcolors)
plotLatt!(ax,Latt,[0,0];site = true, sitelabel = false, linewidth = 3, bond = true)
Colorbar(fig[1,2], colormap = colorschemes[:seismic],colorrange = (-1,1),label = L"\langle W_p\rangle")

resize_to_layout!(fig)
display(fig)
# lsχ

save("$(figurename)/flux_poly_$(L)_$(D)_$(params_Kitaev).png",fig)
save("$(figurename)/flux_poly_$(L)_$(D)_$(params_Kitaev).pdf",fig)
end


