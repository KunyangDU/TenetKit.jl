using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit
include("../analysis/analysis.jl")
include("model.jl")
dataname1 = "../codes/examples/YuanYao/data/spin1"
dataname = "../codes/examples/YuanYao/data"

lsLx = 40:10:120

Ly = 1
Do = 64
D1o = 81
D = 256
D1 = 243
PBC = false
params = (J₁ = 1.0, J₂ = 0.0)

lsU2 = zeros(length(lsLx))
lsU21 = zeros(length(lsLx))
lsU2o = zeros(length(lsLx))
lsU21o = zeros(length(lsLx))

for (i,Lx) in enumerate(lsLx)
    @load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params)_$(true).jld2" gsdata
    lsU2[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
    @load "$(dataname1)/gsdata_$(Lx)x$(Ly)_$(D1)_$(params)_$(true).jld2" gsdata
    lsU21[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(Do)_$(params)_$(false).jld2" gsdata
    lsU2o[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
    @load "$(dataname1)/gsdata_$(Lx)x$(Ly)_$(D1o)_$(params)_$(false).jld2" gsdata
    lsU21o[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
end


model(x,p) = @. p[1]*x + p[2]
f = curve_fit(model, 1 ./ (lsLx), lsU2,randn(2))
f1 = curve_fit(model, 1 ./ (lsLx), lsU21,randn(2))
fo = curve_fit(model, 1 ./ (lsLx), lsU2o,randn(2))
f1o = curve_fit(model, 1 ./ (lsLx), lsU21o,randn(2))

xc = range(0,0.1,10)
yc = model(xc,f.param)
yc1 = model(xc,f1.param)
yco = model(xc,fo.param)
yc1o = model(xc,f1o.param)

figsize = (width = 300,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "Heisenberg chain, D = $(D), PBC = $(PBC)",
xlabel = L"1/L",ylabel = L"\langle U ^2\rangle")

lines!(ax, xc, yc;color = :grey)
lines!(ax, xc, yco;color = :grey)
lines!(ax, xc, yc1o;color = :grey)
lines!(ax, xc, yc1;color = :grey)

scatter!(ax, 1 ./ (lsLx), lsU2; markersize = 14, color = :white, strokewidth = 2, strokecolor = :red,
label = "S=1/2, PBC")
scatter!(ax, 1 ./ (lsLx), lsU21; markersize = 14, color = :white, strokewidth = 2, strokecolor = :blue,
label = "S=1, PBC")
scatter!(ax, 1 ./ (lsLx), lsU2o;color = :red,
label = "S=1/2, OBC")
scatter!(ax, 1 ./ (lsLx), lsU21o;color = :blue,
label = "S=1, OBC")

ylims!(ax,-0.2,1)
xlims!(ax,0,0.03)

Legend(fig[1,2],ax)

resize_to_layout!(fig)
display(fig)


save("YuanYao/figures/gs scaling_AFMH_total_$(Ly)_$(params)_$(D)_$(PBC).pdf",fig)
save("YuanYao/figures/gs scaling_AFMH_total_$(Ly)_$(params)_$(D)_$(PBC).png",fig)


