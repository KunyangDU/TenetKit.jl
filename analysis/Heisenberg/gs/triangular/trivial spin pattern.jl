using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../../analysis/analysis.jl")
include("../../model.jl")

dataname = "../codes/examples/Heisenberg/data/triangular"

D = 128
Lx = 8
Ly = 6
params = (J=1, Δ =1, hx = 0.001 * sqrt(3)/2, hy = 0.001 * 1/2)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

Sx = real.([gsdata[("Sx",)][(i,)] for i in 1:size(Latt)])
Sy = real.([gsdata[("Sy",)][(i,)] for i in 1:size(Latt)])
Sz = real.([gsdata[("Sz",)][(i,)] for i in 1:size(Latt)])

# Sz = [gsdata["Sz"][(i,)] for i in 1:size(Latt)]


fig = Figure()
ax = Axis(fig[1,1],autolimitaspect = true)


plotLatt!(ax,Latt,[0,1];site=true,sitelabel = false,sitesize = 8 * ones(size(Latt)))
# Colorbar(fig[1,2], limits = extrema(Sz), colormap = colorschemes[:bwr])

intensity = 0.6 / maximum(sqrt.(Sx.^2 + Sy.^2 + Sz.^2))
colors = get(colorschemes[:bwr],Sz,extrema(Sz))
for i in 1:size(Latt)
    arrowc!(ax,coordinate(Latt,i)...,intensity*Sx[i],intensity*Sy[i];color = colors[i],linewidth = 4)
end

resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/trivial_spin pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Heisenberg/figures/trivial_spin pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
# Sz
