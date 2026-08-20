using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("../model.jl")

trivialname = "../codes/examples/Heisenberg/data/trivial"
dataname = "Heisenberg/data"
D = 200
params = (J=1.0, Δ = 1.0, Hz = 1.0)

figsize = (width = 400, height = 200)

fig = Figure()
ax = Axis(fig[1,1];
yscale = log10,
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
axE = Axis(fig[2,1];
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
figsize...)
cuttail = 0
fitlength = 10
plot_x = 10 .^ range(-2.,-1.,10)
edgepoints = vcat(1:2, size(Latt)-1:size(Latt))

for Lx in [8,12,16,20]
Ly = 4
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt



@load "$(trivialname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(trivialname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
@load "$(trivialname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE

lsβ2 = 2lsβ[2:end]

lsI2 = map(lsI) do I
    # @show keys(I)
    tmp = 0.0
    ks = keys(I)
    for ky in ks
        validkeys = filter(x -> (x[1][1] ∉ edgepoints) , keys(I[ky]))
        # validkeys = keys(I)
        tmp += sum([I[ky][k] for k in validkeys]) / size(Latt)
    end
    tmp
end .* lsβ2 * 2

# model(x,p) = @. p[1] * x ^ p[2]
# fit_x = 1 ./ lsβ2[end-cuttail-fitlength:end-cuttail]
# fit_y = abs.(lsI2)[end-cuttail-fitlength:end-cuttail]
# fit = curve_fit(model,fit_x,fit_y,[0.1,1.])
# @show fit.param
# @show fit.param,fit_x
scatterlines!(ax,1 ./ lsβ2,abs.(lsI2))
scatterlines!(axE,1 ./ lsβ2[1:end-1], - diff(lsE) ./ diff(lsβ2) .* (lsβ2[1:end-1] .^ 2) / size(Latt),label = "L = $(size(Latt))")
end
# lines!(ax,plot_x, model(plot_x,fit.param);color=:red, linewidth = 2, linestyle = :dash)
axislegend(axE,position = :rb)

# end
for ax in [ax,axE]
    xlims!(ax,1 / 200,10^(-0.))
end
ylims!(ax,10^(-4.),10^(-0.))
ylims!(axE,0,0.15)


resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/SSE_size_$(Lx)x$(Ly)_$(D).png",fig)
save("Heisenberg/figures/SSE_size_$(Lx)x$(Ly)_$(D).pdf",fig)
