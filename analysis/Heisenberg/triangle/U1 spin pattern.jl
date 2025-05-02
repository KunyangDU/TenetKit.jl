using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Heisenberg/data/triangle"

D = 2^7
Lx = 6
Ly = 6
params = (Jz=1,Jxy=0.5,h=0)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

Sz = [gsdata["Sz"][(i,)] for i in 1:size(Latt)]

Sm = maximum(Sz)

figsize = (width = 60*Lx,height = 60*(Ly)*sqrt(3)/2)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
xticks = (0.5 .+ (1:Lx),string.(1:Lx)),yticks = (sqrt(3)/2*(1:Ly),string.(1:Ly)),
title = "$(2Lx)x$(2Ly) ZZ-HC-CY, D=$(D)")
plotLatt!(ax,Latt,[0,sqrt(3)/2];site = true,sitelabel = false)


intensity = 0.3 / maximum(sum(abs.(Sz))/length(Sz))
colors = get(colorschemes[:bwr],Sz,(-Sm,Sm))
for i in 1:size(Latt)
    arrowc!(ax,coordinate(Latt,i)...,0,intensity*Sz[i];color = colors[i],linewidth = 2)
end

Colorbar(fig[1,2],limits = (-Sm,Sm),colormap = :bwr,label = L"\langle S_y\rangle")

# axx = Axis(fig[1,1];autolimitaspect = true,figsize...)
# axy = Axis(fig[2,1];autolimitaspect = true,figsize...)
# axz = Axis(fig[3,1];autolimitaspect = true,figsize...)

# plotLatt!(axx,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sx,(-Sm,Sm)))
# plotLatt!(axy,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sz,(-Sm,Sm)))
# plotLatt!(axz,Latt,[0,1];site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sz,(-Sm,Sm)))

# Colorbar(fig[1,2],limits = (-Sm,Sm),colormap = :bwr)
# Colorbar(fig[2,2],limits = (-Sm,Sm),colormap = :bwr)
# Colorbar(fig[3,2],limits = (-Sm,Sm),colormap = :bwr)

resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/spin pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Heisenberg/figures/spin pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)