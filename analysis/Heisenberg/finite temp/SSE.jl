using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("../model.jl")

trivialname = "../codes/examples/Heisenberg/data/trivial"
dataname = "Heisenberg/data"
D = 2^5
Lx = 50
Ly = 1
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt

figsize = (width = 400, height = 250)

fig = Figure()
ax = Axis(fig[1,1];yscale = log10,xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "$(Ly)x$(Lx) Heisenberg model, D = $(D)", 
xlabel = L"T",
ylabel = L"|\tilde{I}_2|",
figsize...)

cuttail = 190
fitlength = 35
plot_x = 10 .^ range(-2.,-1.,10)
edgepoints = vcat(1:1, size(Latt)-1:size(Latt))

lsH = [1,2]

labels = [
    L"B = 1", L"B = B_c"
]

for (i,H) in enumerate(lsH)
params = (J=1,H=H)
@load "$(trivialname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(trivialname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
lsβ2 = 2lsβ[2:end]

lsI2 = map(lsI) do I
    # @show keys(I)
    validkeys = filter(x -> (x[1][1] ∉ edgepoints) , keys(I))
    # validkeys = keys(I)
    sum([I[k] for k in validkeys]) / length(validkeys)
end .* lsβ2 * 2

model(x,p) = @. p[1] * x ^ p[2]
fit_x = 1 ./ lsβ2[end-cuttail-fitlength:end-cuttail]
fit_y = abs.(lsI2)[end-cuttail-fitlength:end-cuttail]
fit = curve_fit(model,fit_x,fit_y,[0.1,1.])
@show fit.param
# @show fit.param,fit_x
scatterlines!(ax,1 ./ lsβ2[1:end-cuttail],abs.(lsI2)[1:end-cuttail],label = labels[i])

lines!(ax,plot_x, model(plot_x,fit.param);color=:red, linewidth = 2, linestyle = :dash)
end

resize_to_layout!(fig)
# end
xlims!(ax,10^(-2),10^(-0.))
ylims!(ax,10^(-3.),10^(-0.))

axislegend(ax,position = :lt)

display(fig)

save("Heisenberg/figures/SSE_$(Lx)x$(Ly)_$(D).png",fig)
save("Heisenberg/figures/SSE_$(Lx)x$(Ly)_$(D).pdf",fig)
