using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("model.jl")

typename = "trivial"
dataname = "../codes/examples/Heisenberg/data/triangle/$(typename)"
D = 100
Lx = 6
Ly = 6
# params = (J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)
# params = (hy = 1.0, J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)
params = (J = 1,)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

# nb = vcat(neighbor(Latt),neighbor(Latt;level =2),neighbor(Latt;level =3))
nb = neighbor(Latt)
# xbonds,ybonds,zbonds = getxyzbonds(Latt)
ssx = [gsdata["SxSx"][pair] for pair in nb]
ssy = [gsdata["SySy"][pair] for pair in nb]
ssz = [gsdata["SzSz"][pair] for pair in nb]

figsize = (width = 60*Lx,height = 60*(Ly)*sqrt(3)/2)
fig = Figure()
axx = Axis(fig[1,1];autolimitaspect = true,figsize...,
xticks = (0.5 .+ (1:Lx),string.(1:Lx)),yticks = (sqrt(3)/2*(1:Ly),string.(1:Ly)),

title = "$(Ly)x$(Lx) ZZ-HC-CY, D=$(D)")

axy = Axis(fig[2,1];autolimitaspect = true,figsize...,
xticks = (0.5 .+ (1:Lx),string.(1:Lx)),yticks = (sqrt(3)/2*(1:Ly),string.(1:Ly)),
)
axz = Axis(fig[3,1];autolimitaspect = true,figsize...,
xticks = (0.5 .+ (1:Lx),string.(1:Lx)),yticks = (sqrt(3)/2*(1:Ly),string.(1:Ly)),
)

plotLatt!(axx,Latt,[0,sqrt(3)/2];site = true,sitelabel = false,bond = false)
plotLatt!(axy,Latt,[0,sqrt(3)/2];site = true,sitelabel = false,bond = false)
plotLatt!(axz,Latt,[0,sqrt(3)/2];site = true,sitelabel = false,bond = false)

limits = (-0.1,0.1)

plotbond!(axx,Latt,nb,ssx,[0,sqrt(3)/2];colorlimit = limits,linewidth = 40,colormap = :bwr)
plotbond!(axy,Latt,nb,ssy,[0,sqrt(3)/2];colorlimit = limits,linewidth = 40,colormap = :bwr)
plotbond!(axz,Latt,nb,ssz,[0,sqrt(3)/2];colorlimit = limits,linewidth = 40,colormap = :bwr)

Colorbar(fig[1,2],limits = limits,colormap = :bwr,label = L"\langle S_xS_x\rangle")
Colorbar(fig[2,2],limits = limits,colormap = :bwr,label = L"\langle S_yS_y\rangle")
Colorbar(fig[3,2],limits = limits,colormap = :bwr,label = L"\langle S_zS_z\rangle")

hidexdecorations!(axx,grid = false,ticks = false)
hidexdecorations!(axy,grid = false,ticks = false)

resize_to_layout!(fig)
display(fig)

save("Heisenberg/triangle/figures/$(typename)/spin corr pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Heisenberg/triangle/figures/$(typename)/spin corr pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)




