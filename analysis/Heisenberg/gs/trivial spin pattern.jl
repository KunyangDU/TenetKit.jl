using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/data/trivial"

D = 128
Lx = 8
Ly = 6
params = (J=1, Δ =1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

Sz = real.([gsdata[("Sz",)][(i,)] for i in 1:size(Latt)])

# Sz = [gsdata["Sz"][(i,)] for i in 1:size(Latt)]

figsize = (height = 50 * 7, width = 50 * 8)

fig = Figure()
ax = Axis(fig[1,1],autolimitaspect = true;figsize...,
xticks = 1:20,yticks = 1:20,
xgridvisible = false,ygridvisible = false)


plotLatt!(ax,Latt,[0,1];site=true,sitelabel = false,sitesize = 8 * ones(size(Latt)))
# Colorbar(fig[1,2], limits = extrema(Sz), colormap = colorschemes[:bwr])

intensity = 0.6 / maximum(norm.(Sz))
colors = get(colorschemes[:bwr],Sz,extrema(Sz))
for i in 1:size(Latt)
    arrowc!(ax,coordinate(Latt,i)...,0,intensity*Sz[i];color = colors[i],linewidth = 4)
end

Colorbar(fig[1,2], colorrange = (-0.5,0.5),colormap = :bwr,label = L"\langle S_z\rangle")

resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/trivial_spin pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Heisenberg/figures/trivial_spin pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
# Sz
