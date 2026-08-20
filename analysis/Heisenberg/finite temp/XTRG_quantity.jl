using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("../model.jl")


dataname = "../codes/examples/Heisenberg/data/XTRG"

Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
L = size(Latt)

D = 300
DS = 32
algetol = 1e-4
SETTNtol = 1e-12

N = 15
νs = [convert(Float64,i//4) for i in 0]

params = (J = 1.0, Δ = 1.0, Hz = 0.0)

figsize = (width = 400, height = 200)

fig = Figure()
axE = Axis(fig[1,1];
# yscale = log10,å
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "XTRG $(Ly)x$(Lx) Heisenberg model\n$(params)", 
xlabel = L"T",
ylabel = L"E/L",
figsize...)
axS = Axis(fig[2,1];
# yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
xlabel = L"T",
ylabel = L"S/L/\ln 2",
figsize...)

axC = Axis(fig[3,1];
# yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
xlabel = L"T",
ylabel = L"C/L",
figsize...)


lsE = zeros(length(νs),N)
lsF = zeros(length(νs),N)
lsβtotal = zeros(length(νs),N)
for (iν, ν) in enumerate(νs)
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_$(ν).jld2" lsβ
lsβtotal[iν,:] = lsβ
for i in eachindex(lsβ)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(DS)_$(algetol)_$(SETTNtol)_$(params)_$(ν)_$(i).jld2" data
    lsE[iν,i] = data["E"] / size(Latt)
    lsF[iν,i] = data["F"] / size(Latt)
end
end
lsβtotal = lsβtotal[:]
lsE = lsE[:]
lsF = lsF[:]
lsC = -diff(lsE) ./ diff(log.(lsβtotal)) .* lsβtotal[2:end]
scatterlines!(axE, 1 ./ lsβtotal, lsE)
scatterlines!(axS, 1 ./ lsβtotal, (lsE .- lsF) .* lsβtotal / log(2))
scatterlines!(axC, 1 ./ lsβtotal[2:end], lsC)

for ax in [axS,axE,axC]
xlims!(ax,1e-2,1e1)
end
# ylims!(axF,1e-11,1e-7)
ylims!(axS,0,1)

hidexdecorations!(axE,ticks = false,minorticks = false)


resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/XTRG_quantity_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/figures/XTRG_quantity_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
