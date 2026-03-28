using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/ZZHC"
figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC/flux"
tailname = ""

D = 256
Lx = 3
Ly = 4

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt



# for Hf in 0:0.08:0.8
Hf = 0.0
θ = 0.0
ϕ = 0.0

K = 1.0
# Γ = 0.0
for Γ′ in 0:0.02:0.4
params1_Kitaev = (K = K, Γ′ = Γ′)

# Hf = 0.8


# for (i,Hf) in enumerate(lsHf)


Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)
# Hcx,Hcy,Hcz = Hx * [1,-1,0] + Hy * [1,1,-2] + Hz * [1,1,1]

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

# println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata

obsdata = gsdata["obs"]

flux_shift = [2*sqrt(3)/3,0]
flux_Latt = YCTria(2Lx-1,Ly)
direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]
flux_Latt_sites,fluxdirections,_ = getPBCflux(Latt,flux_Latt,[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]];d = 1/sqrt(3),edge_shift = [0,1],flux_shift = [2*sqrt(3)/3,0])

fluxsites = []
fluxvalues = []
flux_key = filter(x -> length(x[1]) == 6, collect(keys(obsdata)))
for kdc in flux_key, k in keys(obsdata[kdc])
    ind = findfirst(x -> isequal(x,k[1]),flux_Latt_sites)
    push!(fluxsites,collect(coordinate(flux_Latt,ind)) + flux_shift)
    push!(fluxvalues,obsdata[kdc][k])
end
# flux_key
# fluxvalues
figsize = (height = 300,width = 300)
@show sum(fluxvalues * 2^6)

fluxcolors = get(colorschemes[:seismic],fluxvalues * 2^6,(-1,1))
fig = Figure()
ax = Axis(fig[1,1],autolimitaspect = true;figsize...,
title = "size = $(size(Latt)), D = $(D), K = $(K), Γ′ = $(Γ′)\n $((Hx,Hy,Hz))")

# @show fluxcolors
polyHexagon!(ax,fluxsites,fluxcolors;scale = 1/sqrt(3),rot_mat = [0 -1;1 0])
plotLatt!(ax,Latt,[0,1];site = true, sitelabel = false, linewidth = 3, bond = true)
Colorbar(fig[1,2], colormap = colorschemes[:seismic],colorrange = (-1,1),label = L"\langle W_p\rangle")

resize_to_layout!(fig)
display(fig)

save("$(figurename)/flux_poly_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).png",fig)
save("$(figurename)/flux_poly_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).pdf",fig)
end


