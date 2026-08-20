using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("../model.jl")


dataname = "../codes/examples/Heisenberg/data/XTRG"
edname = "../codes/examples/Heisenberg/data/ed"

Lx = 8
Ly = 1
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
L = size(Latt)

D = 64
DS = 2^4

params = (J = 1.0, Δ = 1.0, Hz = 0.0)

@load "$(edname)/lsdata_$(Lx)x$(Ly)_$(params).jld2" lsdata
lsdataed = lsdata 

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_1.jld2" lsβ

@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params)_1.jld2" lsF
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params)_1.jld2" lsE
lsF1 = lsF
lsE1 = lsE

@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params)_2.jld2" lsF
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params)_2.jld2" lsE
lsF2 = lsF
lsE2 = lsE

finddata(dicts::Vector,name::String) = map(x -> x[name],dicts)
lsFed = real.(finddata(lsdataed,"F"))
lsEed = real.(finddata(lsdataed,"E"))

lsFerr1 = abs.((lsFed .- lsF1) ./ lsFed)
lsFerr2 = abs.((lsFed .- lsF2) ./ lsFed)
lsEerr1 = abs.((lsEed .- lsE1) ./ lsEed)
lsEerr2 = abs.((lsEed .- lsE2) ./ lsEed)


figsize = (width = 300, height = 200)

fig = Figure()
axF = Axis(fig[1,1];
yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "XTRG $(Ly)x$(Lx) Heisenberg model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"|(F_{ED}-F)/F_{ED}|",
figsize...)
axE = Axis(fig[2,1];
yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
xlabel = L"T",
ylabel = L"|(E_{ED}-E)/E_{ED}|",
figsize...)

scatterlines!(axF, 1 ./ lsβ, lsFerr1, label = "1 site + CBE")
scatterlines!(axF, 1 ./ lsβ, lsFerr2, label = "2 site")

scatterlines!(axE, 1 ./ lsβ, lsEerr1, label = "1 site + CBE")
scatterlines!(axE, 1 ./ lsβ, lsEerr2, label = "2 site")

for ax in [axF,axE]
xlims!(ax,1e-2,1e1)
end
ylims!(axF,1e-11,1e-7)
ylims!(axE,1e-13,1e-6)

hidexdecorations!(axF,ticks = false,minorticks = false)

axislegend(axF,position = :lb)

resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/XTRG_ED_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/figures/XTRG_ED_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
