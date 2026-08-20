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

indβ = 40
validinds = 2:46

params = (J = 1.0, Δ = 1.0, Hz = 0.1)

figsize = (width = 400, height = 250)

fig = Figure()

axI = Axis(fig[1,1];
# yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "Heisenberg model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"|\tilde{I}_2|",
figsize...)

ax = Axis(fig[2,1];
yscale = log10,
# xscale = log10,
xlabel = L"column",
ylabel = L"\tilde{I}_2",
figsize...)


for Lx in [12,16,20]

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt


edgepoints = vcat(1:2Ly, size(Latt)-2Ly:size(Latt))
# edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
# lsβ2 = vcat(lsβ2,42:2:200)
lsI = zeros(size(Latt), length(validinds))
lsβeff = zeros(length(validinds))
for (ii,i) in enumerate(validinds)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    I = data["I"]
    ks = keys(I)
    for j in 1:size(Latt)
        for k in ks
            validkeys = filter(x -> (x[1][1] == j) , keys(I[k]))
            isempty(validkeys) && continue
            lsI[j,ii] += sum([I[k][s] for s in validkeys]) / size(Latt) * lsβ2[i] * 2
        end
    end
    lsβeff[ii] = lsβ2[i]
end

scatterlines!(axI,1 ./ lsβeff,(sum(lsI[2Ly:end-2Ly + 1,:],dims = 1)[:]),label = "$(Lx)x$(Ly)")

# scatterlines!(ax,(1:div(size(Latt),2)-Ly)/div(size(Latt),2),(lsI[1:div(size(Latt),2) - Ly,indβ]) |> x -> x .- minimum(lsI[:,indβ]))
scatterlines!(ax,1:div(Lx,2),sum(reshape(lsI[:,indβ],Ly,:),dims = 1)[:][1:div(Lx,2)] .- 4 * minimum(lsI[:,indβ]))

if Lx == 12
    lines!(axI,1 ./ lsβeff[indβ] * [1,1],0.1 * [-1,1])
end
end

xlims!(axI,10^(-1.75),10^(-0.))
# xlims!(ax,0,1)

axislegend(axI,position = :rb)

ylims!(axI,-0.08,0.06)

resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/SSE_step_onsite_size_$(D)_$(params).pdf",fig)
save("Heisenberg/figures/SSE_step_onsite_size_$(D)_$(params).png",fig)
