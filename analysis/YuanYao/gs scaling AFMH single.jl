using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit
include("../analysis/analysis.jl")
include("model.jl")
# dataname1 = "../codes/examples/YuanYao/data/spin1"
dataname = "../codes/examples/YuanYao/data"

Lx = 300

Ly = 1
D = 64
PBC = false
params = (J₁ = 1.0, J₂ = 0.0)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params)_$(PBC).jld2" gsdata
lsL = 100:50:size(Latt)
lsU2 = zeros(length(lsL))

for (i,L) in enumerate(lsL)
    sites = collect(div(size(Latt)-L,2)+1:L+div(size(Latt)-L,2))
    lsU2[i] = gsdata[(Tuple(["Sn" for i in sites]),)][(Tuple([i for i in sites]),)]
end
# lsU2


# model(x,p) = @. p[1]*x + p[2]
# f = curve_fit(model, 1 ./ (lsLx), lsU2,randn(2))
# f1 = curve_fit(model, 1 ./ (lsLx), lsU21,randn(2))

# xc = range(0,0.1,10)
# yc = model(xc,f.param)
# yc1 = model(xc,f1.param)

figsize = (width = 300,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "Heisenberg chain, D = $(D), PBC = $(PBC)",
xlabel = L"1/L",ylabel = L"\langle U ^2\rangle")

# lines!(ax, xc, yc;color = :grey)
# lines!(ax, xc, yc1;color = :grey)

scatter!(ax, 1 ./ (lsL), lsU2;color = :red, label = L"S=1/2")
# scatter!(ax, 1 ./ (lsLx), lsU21;color = :blue, label = L"S=1")

# ylims!(ax,0.0,1)
xlims!(ax,0,0.009)

axislegend(ax,position = :rc)

resize_to_layout!(fig)
display(fig)


# save("YuanYao/figures/gs scaling_AFMH_total_$(Ly)_$(params)_$(D)_$(PBC).pdf",fig)
# save("YuanYao/figures/gs scaling_AFMH_total_$(Ly)_$(params)_$(D)_$(PBC).png",fig)


