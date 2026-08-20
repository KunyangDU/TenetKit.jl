using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/data/trivial"

Lx = 64
Ly = 1
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 64
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 100

params = (J = 1.0, Δ = 1.0, Hz = 1.0)

edgepoints = vcat(1:2Ly, size(Latt)-2Ly:size(Latt))
# edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
# lsβ2 = vcat(lsβ2,42:2:200)
validinds = 2:100
lsE = zeros(length(validinds))
lsF = zeros(length(validinds))
lsI = zeros(length(validinds))

lsβeff = zeros(length(validinds))
for (ii,i) in enumerate(validinds)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    lsE[ii] = data["E"] / size(Latt)
    lsF[ii] = data["F"] / size(Latt)
    I = data["I"]
    ks = keys(I)
    for k in ks
        validkeys = filter(x -> (x[1][1] ∉ edgepoints) , keys(I[k]))
        lsI[ii] += sum([I[k][s] for s in validkeys]) / size(Latt) * lsβ2[ii]
    end
    lsβeff[ii] = lsβ2[ii]
end

figsize = (width = 400, height = 200)

fig = Figure()
axI = Axis(fig[1,1];
yscale = log10,
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
xticks = 10. .^ (-2:1:1),
figsize...)
axC = Axis(fig[2,1];
# yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
# title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"C/L",
xticks = 10. .^ (-2:1:1),
figsize...)

axS = Axis(fig[3,1];
# yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
# title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"S/L",
xticks = 10. .^ (-2:1:1),
figsize...)

scatterlines!(axI,1 ./ lsβeff,abs.(lsI))
scatterlines!(axC,1 ./ lsβeff[2:end],- diff(lsE) ./ diff(lsβeff) .* lsβeff[2:end] .^ 2)
scatterlines!(axS,1 ./ lsβeff,(lsE .- lsF) .* lsβeff / log(2))

for ax in [axI,axC,axS]
xlims!(ax,10^(-2),10^(1.))
# ylims!(ax,10^(-5.),10^(0.))
end
ylims!(axS,0,1)

hidexdecorations!(axI,ticks = false,minorticks = false)
hidexdecorations!(axC,ticks = false,minorticks = false)

resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
