using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/XTRG/ZZHC"

Lx = 4
Ly = 3
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 256
DS = 2^4
lsν = [0.0,0.5]
N = 20

figsize = (width = 400, height = 200)

fig = Figure()
ax = Axis(fig[1,1];
yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
xticks = [0.001,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D), K = $(params.K)", 
xlabel = L"T",
ylabel = L"|\tilde{I}_2|",
figsize...)

for Hc in [0.1,0.2,0.3]
    params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = Hc)
lsI,lsβeff = let

lsI = zeros(length(lsν),N)
lsβeff = zeros(length(lsν),N)
# edgepoints = vcat(1:4Ly,size(Latt)-4Ly+1:size(Latt))
edgepoints = []
for (iν,ν) in enumerate(lsν)
# edgepoints = vcat(1:4Ly + 2, size(Latt)-4Ly - 1:size(Latt))
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_$(ν).jld2" lsβ

for i in 1:N
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(ν)_$(i).jld2" data
    I = data["SSE"]
    ks = keys(I)
    for k in ks
        validkeys = filter(x -> (x[1][1] ∉ edgepoints) , keys(I[k]))
        lsI[iν,i] += real(sum([I[k][s] for s in validkeys]) / size(Latt) * lsβ[i])
    end
    lsβeff[iν,i] = lsβ[i]
end
# @show lsI
end 
lsI[:],lsβeff[:]
end


scatter!(ax,1 ./ lsβeff,abs.(lsI);
color = :white, markersize = 12, strokewidth = 2,strokecolor = map(x -> x > 0 ? :black : :red, lsI))

scatterlines!(ax,1 ./ lsβeff,abs.(lsI),label = "Hc = $(Hc)")
end
xlims!(ax,1 / 4096,10^(-0.))
ylims!(ax,10^(-4.),10^(-0.5))

axislegend(ax,position = :rb)

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/SSE_single_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
