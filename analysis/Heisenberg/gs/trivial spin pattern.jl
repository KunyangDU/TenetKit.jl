using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/data/trivial"

D = 2^7
Lx = 4
Ly = 4
params = (J=1,h=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

Sz = [gsdata["Sz"][(i,)] for i in 1:size(Latt)]


fig = Figure()
ax = Axis(fig[1,1])


plotLatt!(ax,Latt,[0,1];site=true,sitelabel = false,sitecolor = get(colorschemes[:bwr],Sz,:extrema))
Colorbar(fig[1,2], limits = extrema(Sz), colormap = colorschemes[:bwr])
resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/trivial_spin pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Heisenberg/figures/trivial_spin pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

