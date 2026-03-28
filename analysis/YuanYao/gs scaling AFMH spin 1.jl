using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit
include("../analysis/analysis.jl")
include("model.jl")
dataname = "../codes/examples/YuanYao/data/spin1"

lsLx = 100:50:400

Ly = 1
D = 64
PBC = false
params = (J₁ = 1.0, J₂ = 0.0)

lsU2 = zeros(length(lsLx))
for (i,Lx) in enumerate(lsLx)
    @load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params)_$(PBC).jld2" gsdata
    lsU2[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
end

lsU2

model(x,p) = @. p[1]*x + p[2]
f = curve_fit(model, 1 ./ (lsLx), lsU2,randn(2))

xc = range(0,0.1,10)
yc = model(xc,f.param)
figsize = (width = 300,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "D = $(D), PBC = $(PBC)",
xlabel = L"1/\mathrm{ln}L",ylabel = L"\langle U ^2\rangle")

lines!(ax, xc, yc;color = :grey)
scatter!(ax, 1 ./ (lsLx), lsU2;color = :red)
ylims!(ax,0.0,1)
xlims!(ax,0,0.009)

resize_to_layout!(fig)
display(fig)


save("YuanYao/figures/gs scaling_$(Ly)_$(params)_$(D)_$(PBC).pdf",fig)
save("YuanYao/figures/gs scaling_$(Ly)_$(params)_$(D)_$(PBC).png",fig)


