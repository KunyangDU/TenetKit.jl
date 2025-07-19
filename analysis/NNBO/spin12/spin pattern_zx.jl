using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/NNBO/spin12/data/"
figurename = "NNBO/spin12/figures"

D = 2^7
Lx = 4
Ly = 4
params1_1_Kitaev = (J1 = -1, K1 = 0.6, Γ1 = 0, Γ1′ = 0)
params1_3_JDH = (J3 = 1, D = 3)
paramsh = (h=0.0, H = 0.)

params1_cry = _Cub2Cry(params1_1_Kitaev)
params = merge(params1_cry,params1_3_JDH,paramsh)
params_Kitaev = merge(params1_1_Kitaev,params1_3_JDH,paramsh)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg

Sx = [gsdata["Sx"][(i,)] for i in 1:size(Latt)]
Sy = [gsdata["Sy"][(i,)] for i in 1:size(Latt)]
Sz = [gsdata["Sz"][(i,)] for i in 1:size(Latt)]

Sm = maximum(sqrt.((Sx.^2 + Sy.^2 + Sz.^2))[4Ly+1:size(Latt)-4Ly-1])

figsize = (width = Lx*sqrt(3)*50,height = Ly*60)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (1:0.5:Ly+0.5,string.(1:2Ly)),
title = "$(2Lx)x$(Ly)x2 ZZ-HC-CY, xz, D=$(D)")
plotLatt!(ax,Latt,[0,1];site = false,sitelabel = false)


intensity = 0.45 / maximum(sum(sqrt.(Sz.^2 + Sx.^2))/length(Sy))
colors = get(colorschemes[:bwr],Sz,(-Sm,Sm))
for i in 1:size(Latt)
    arrowc!(ax,coordinate(Latt,i)...,intensity*Sx[i],intensity*Sz[i];color = colors[i],linewidth = 2)
end

Colorbar(fig[1,2],limits = (-Sm,Sm),colormap = :bwr,label = L"\langle S_y\rangle")

# axx = Axis(fig[1,1];autolimitaspect = true,figsize...)
# axy = Axis(fig[2,1];autolimitaspect = true,figsize...)
# axz = Axis(fig[3,1];autolimitaspect = true,figsize...)

# plotLatt!(axx,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sx,(-Sm,Sm)))
# plotLatt!(axy,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sy,(-Sm,Sm)))
# plotLatt!(axz,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sz,(-Sm,Sm)))

# Colorbar(fig[1,2],limits = (-Sm,Sm),colormap = :bwr)
# Colorbar(fig[2,2],limits = (-Sm,Sm),colormap = :bwr)
# Colorbar(fig[3,2],limits = (-Sm,Sm),colormap = :bwr)

resize_to_layout!(fig)
display(fig)
save("$(figurename)/spin pattern_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_xy.png",fig)
save("$(figurename)/spin pattern_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_xy.pdf",fig)
