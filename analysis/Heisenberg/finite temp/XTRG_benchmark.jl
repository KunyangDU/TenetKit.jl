using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("../model.jl")


dataname = "../codes/examples/Heisenberg/data/XTRG"
edname = "../codes/examples/Heisenberg/data/ed"

Lx = 6
Ly = 1
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
L = size(Latt)

ν = 0.0

D = 64
DS = 32
algetol = 1e-4
SETTNtol = 1e-12

params = (J = 1.0, Δ = 1.0, Hz = 0.0)

@load "$(edname)/lsdata_$(Lx)x$(Ly)_$(params).jld2" lsdata
lsdataed = lsdata 

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_$(ν).jld2" lsβ
lsdata = []
for i in eachindex(lsβ)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(DS)_$(algetol)_$(SETTNtol)_$(params)_$(ν)_$(i).jld2" data
    push!(lsdata,data)
end

finddata(dicts::Vector,name::String) = map(x -> x[name],dicts)
lsFed = real.(finddata(lsdataed,"F"))
lsEed = real.(finddata(lsdataed,"E"))

lsF = finddata(lsdata,"F")
lsE = finddata(lsdata,"E")

lsFerr = abs.((lsFed .- lsF) ./ lsFed)
lsEerr = abs.((lsEed .- lsE) ./ lsEed)

figsize = (width = 400, height = 200)

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
title = "XTRG $(Ly)x$(Lx) Heisenberg model\n$(params)", 
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

scatterlines!(axF, 1 ./ lsβ, lsFerr, label = "D = $(D), DS = $(DS), altol = $(algetol), SNtol = $(SETTNtol)")

scatterlines!(axE, 1 ./ lsβ, lsEerr, label = "1 site + CBE")

for ax in [axF,axE]
xlims!(ax,1e-2,1e1)
end
# ylims!(axF,1e-11,1e-7)
# ylims!(axE,1e-13,1e-6)

hidexdecorations!(axF,ticks = false,minorticks = false)

Legend(fig[:,2],axF,position = :lb)

resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/XTRG_BENCHMARK_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/figures/XTRG_BENCHMARK_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
