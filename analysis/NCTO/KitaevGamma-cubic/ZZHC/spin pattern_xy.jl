using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/BCAO/arxiv2025/data/Hx"
figurename = "BCAO/arxiv2025/figures/Hx"
tailname = ""

D = 2^8
Lx = 3
Ly = 6

params1_Kitaev = (J1 = -0.59, K1 = -1.0, Γ1 = 0.53, Γ1′ = 0.11)
params23 = (J2 = -0.038, J3xy = 0.31, J3z = 0.0092, Hx = 0.12)
paramsh = (pinh=0.,)
# for J3 in 0.3:0.01:0.4
# params1_Kitaev = (J1 = -0.59, K1 = 0.0, Γ1 = 0.0, Γ1′ = 0.0)
# params23 = (J2 = -0.038, J3xy = J3, J3z = 0.0)
# paramsh = (pinh=0.,)

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params23,paramsh)
params_Kitaev = merge(params1_Kitaev,params23,paramsh)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)$(tailname).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)$(tailname).jld2" lsEg


Sx = [gsdata[(("Sx",),)][((i,),)] for i in 1:size(Latt)]
Sy = [gsdata[(("Sy",),)][((i,),)] for i in 1:size(Latt)]
Sz = [gsdata[(("Sz",),)][((i,),)] for i in 1:size(Latt)]

Sm = maximum(sqrt.((Sx.^2 + Sy.^2 + Sz.^2)))

figsize = (width = Lx*sqrt(3)*50,height = Ly*60)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
xticks = (sqrt(3)*11/12 .+ sqrt(3)/2 .* (0:2Lx-1),string.(1:2Lx)),
yticks = (1:0.5:Ly+0.5,string.(1:2Ly)),
title = "$(2Lx)x$(2Ly) ZZ-HC-CY, zy, D=$(D)")
plotLatt!(ax,Latt,[0,1];site = false,sitelabel = false)

intensity = 0.45 / maximum(sum(sqrt.(Sy.^2 + Sx.^2))/length(Sy))
colors = get(colorschemes[:bwr],Sy,(-Sm,Sm))
for i in 1:size(Latt)
    arrowc!(ax,coordinate(Latt,i)...,intensity*(Sx[i] - sum(Sx)/length(Sx)),intensity*Sy[i];color = colors[i],linewidth = 2)
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
# end
lsEg