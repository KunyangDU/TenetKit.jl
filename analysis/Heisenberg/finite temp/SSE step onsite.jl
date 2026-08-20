using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/data/trivial"

Lx = 20
Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 200
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 100

indβ = 40

params = (J = 1.0, Δ = 1.0, Hz = 0.1)

edgepoints = vcat(1:2Ly, size(Latt)-2Ly:size(Latt))
# edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
# lsβ2 = vcat(lsβ2,42:2:200)
validinds = 2:46
lsI = zeros(size(Latt), length(validinds))
lsβeff = zeros(size(Latt), length(validinds))
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
lsI = abs.(lsI)
figsize = (width = 400, height = 250)

fig = Figure()
ax = Axis(fig[1,1];
# yscale = log10,
# xscale = log10,
# xminorticksvisible = true, 
# xminorticks = IntervalsBetween(10),
# yminorticksvisible = true,
# yminorticks = IntervalsBetween(10),
# xgridvisible = false, ygridvisible = false,
# xminorgridvisible = false, yminorgridvisible = false,
# title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
# xlabel = L"T",
# ylabel = L"|\tilde{I}_2|",
figsize...)

# scatterlines!(ax,1 ./ lsβeff,lsI)

scatterlines!(ax,range(0,1,size(Latt)),lsI[:,indβ])

# xlims!(ax,10^(-2),10^(-0.))
# ylims!(ax,10^(-5.),10^(-1.))

resize_to_layout!(fig)
display(fig)
# save("Heisenberg/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
# save("Heisenberg/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
