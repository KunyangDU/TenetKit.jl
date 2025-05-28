using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/BCAO/arxivJKGGp/data/pin"
figurename = "BCAO/arxivJKGGp/figures"
D = 2^6
Lx = 6
Ly = 4
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.59,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=0.0,J3xy=0.0,J3z=0.0)
# params = (J1=-0.1,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.1,K1=-1.0,Γ1=0.54,Γ1′=0.11,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.038,J3xy=0.3,J3z=0.01)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.008)
# params = (J1=-0.50035,K1=-0.85893,Γ1=0.4538,Γ1′=0.09311,J2= -0.0321,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.093,J2=-0.032,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.28,J3z=0.0078)

# 6x4
params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.28,J3z=0.0078)

# params = (J1=-0.514,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.032,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.5,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.036,J3xy=0.26,J3z=0.0078)
# params = (J1=-0.52,K1=-0.86,Γ1=0.45,Γ1′=0.09,J2=-0.038,J3xy=0.26,J3z=0.0078)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

Sx = [gsdata["Sx"][(i,)] for i in 1:size(Latt)]
Sy = [gsdata["Sy"][(i,)] for i in 1:size(Latt)]
Sz = [gsdata["Sz"][(i,)] for i in 1:size(Latt)]

Sm = maximum(vcat(Sx,Sy,Sz))

figsize = (width = Lx*sqrt(3)*50,height = Ly*60)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (1:0.5:Ly+0.5,string.(1:2Ly)),
title = "$(2Lx)x$(Ly)x2 ZZ-HC-CY, zy, D=$(D)")
plotLatt!(ax,Latt,[0,1];site = false,sitelabel = false)


intensity = 0.45 / maximum(sum(sum(sqrt.(Sz.^2 + Sy.^2)))/length(Sy))
colors = get(colorschemes[:bwr],Sy,(-Sm/3,Sm/3))
for i in 1:size(Latt)
    arrowc!(ax,coordinate(Latt,i)...,intensity*Sz[i],intensity*Sy[i];color = colors[i],linewidth = 2)
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
save("$(figurename)/spin pattern_$(Lx)x$(Ly)_$(D)_$(params)_zy.png",fig)
save("$(figurename)/spin pattern_$(Lx)x$(Ly)_$(D)_$(params)_zy.pdf",fig)