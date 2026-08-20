using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/data/trivial"

Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)

D = 200
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 100

params = (J = 1.0, Δ = 1.0, Hz = 0.1)

figsize = (width = 400, height = 250)

fig = Figure()
ax = Axis(fig[1,1];
# yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "$(Ly)x$(Lx) Heisenberg model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"|\tilde{I}_2|",
figsize...)


for Lx in [8,12,16,20]
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt



edgepoints = vcat(1:2Ly, size(Latt)-2Ly:size(Latt))
# edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
# lsβ2 = vcat(lsβ2,42:2:200)
validinds = 2:47
lsI = zeros(length(validinds))
lsβeff = zeros(length(validinds))
for (ii,i) in enumerate(validinds)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    I = data["I"]
    ks = keys(I)
    for k in ks
        validkeys = filter(x -> (x[1][1] ∉ edgepoints) , keys(I[k]))
        lsI[ii] += sum([I[k][s] for s in validkeys]) / size(Latt) * lsβ2[i] * 2
    end
    lsβeff[ii] = lsβ2[i]
end
lsI = (lsI)

scatterlines!(ax,1 ./ lsβeff,lsI)

end
xlims!(ax,10^(-2),10^(-0.))
# ylims!(ax,10^(-5.),10^(-1.))

resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
