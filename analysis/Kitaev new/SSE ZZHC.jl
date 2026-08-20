using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/ZZHC"

Lx = 6
Ly = 3
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 400
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 50

params = (K = -1.0, Ha = 0.0, Hb = 0.0, Hc = 0.1)

edgepoints = vcat(1:4Ly + 2, size(Latt)-4Ly - 1:size(Latt))
# edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
# lsβ2 = vcat(lsβ2,42:2:200)
validinds = 2:46
lsI = zeros(length(validinds))
lsβeff = zeros(length(validinds))
for (ii,i) in enumerate(validinds)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    I = data["SSE"]
    ks = keys(I)
    for k in ks
        validkeys = filter(x -> (x[1][1] ∉ edgepoints) , keys(I[k]))
        lsI[ii] += real(sum([I[k][s] for s in validkeys]) / size(Latt) * lsβ2[ii])
    end
    lsβeff[ii] = lsβ2[ii]
end
# @show lsI
lsI = abs.(lsI)
figsize = (width = 300, height = 200)

fig = Figure()
ax = Axis(fig[1,1];
yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
xticks = [0.01,0.05,0.1,0.5,1.0],
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"|\tilde{I}_2|",
figsize...)

scatterlines!(ax,1 ./ lsβeff,lsI)

xlims!(ax,1 / 200,10^(-0.))
ylims!(ax,10^(-5.),10^(0.))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
