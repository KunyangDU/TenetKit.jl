using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit
include("../analysis/analysis.jl")
include("model.jl")
dataname = "../codes/examples/YuanYao/data"

lsLx = 40:10:120
lsLx1 = 41:10:121

Ly = 1
D = 256
Do = 64
PBC = false
params = (J₁ = 1.0, J₂ = 0.5)

lsU2 = zeros(length(lsLx))
lsU2o = zeros(length(lsLx))
lsU21 = zeros(length(lsLx1))
lsU21o = zeros(length(lsLx1))

for (i,Lx) in enumerate(lsLx)
    @load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params)_$(true).jld2" gsdata
    lsU2[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(Do)_$(params)_$(false).jld2" gsdata
    lsU2o[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
end

for (i,Lx) in enumerate(lsLx1)
    @load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params)_$(true).jld2" gsdata
    lsU21[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(Do)_$(params)_$(false).jld2" gsdata
    lsU21o[i] = gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]
end


model(x,p) = @. p[1]*x + p[2]
f = curve_fit(model, 1 ./ (lsLx), lsU2,randn(2))
f1 = curve_fit(model, 1 ./ (lsLx1), lsU21,randn(2))
fo = curve_fit(model, 1 ./ (lsLx1), lsU2o,randn(2))
f1o = curve_fit(model, 1 ./ (lsLx1), lsU21o,randn(2))

xc = range(0,0.05,10)
yc = model(xc,f.param)
yc1 = model(xc,f1.param)
yco = model(xc,fo.param)
yc1o = model(xc,f1o.param)

figsize = (width = 300,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "MG chain $(params)",
xlabel = L"1/L",ylabel = L"\langle U ^2\rangle")

lines!(ax, xc, yc;color = :grey)
lines!(ax, xc, yc1;color = :grey)
lines!(ax, xc, yc1o;color = :grey)

scatter!(ax, 1 ./ (lsLx), lsU2, markersize = 14, color = :white, strokewidth = 2, strokecolor = :blue,
label = "D = $D, PBC, even")
scatter!(ax, 1 ./ (lsLx1), lsU21, markersize = 14, color = :white, strokewidth = 2, strokecolor = :red,
label = "D = $D, PBC, odd")

scatter!(ax, 1 ./ (lsLx), lsU2o, color = :blue,
label = "D = $Do, OBC even")
scatter!(ax, 1 ./ (lsLx1), lsU21o, color = :red,
label = "D = $Do, OBC, odd")

# axislegend(ax, position = :lc)
Legend(fig[1,2],ax)
ylims!(ax,-0.5,1)
xlims!(ax,0,0.03)

resize_to_layout!(fig)
display(fig)


save("YuanYao/figures/gs scaling_MG_total_$(Ly)_$(params)_$(D)_$(PBC).pdf",fig)
save("YuanYao/figures/gs scaling_MG_total_$(Ly)_$(params)_$(D)_$(PBC).png",fig)


