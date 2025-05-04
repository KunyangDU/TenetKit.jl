
using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

trivialname = "../codes/examples/Heisenberg/data/triangle/trivial"
figurename = "tanTRG/structure factor"

D = 2^9
Lx = 6
Ly = 4
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt


lsH = 1:0.2:3
trivialparams = (J=1,H=lsH[1])
@load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(trivialparams).jld2" lsβ2
Sz = zeros(length(lsβ2),length(lsH))

for (iH,H) in enumerate(lsH)
    trivialparams = (J=1,H=H)
    @load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(trivialparams).jld2" data
    for (iβ,β) in enumerate(lsβ2)
        Szs = [data["obs"][iβ]["Sz"][(i,)] for i in 1:size(Latt)]
        Sz[iβ,iH] = sum(Szs) / size(Latt)
    end
    @show data["E2"]
end
figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];
xlabel = L"H\ /\ J",ylabel = L"3\langle\mathbf{S}\cdot\mathbf{\hat{H}}\rangle\ /\ S",
title = "Magnetization, $(Lx)x$(Ly) ZZ-HC-CY, D=$(D)",
# xticks = 0:0.1:1,
xticks = 0:0.5:10,
yticks = 0:0.5:3,
figsize...)

xlims!(ax,0,5)
ylims!(ax,0,3)
selectedβ = reverse(length(lsβ2):-4:17)
colors = [:blue,:green,:red]
for i in selectedβ
    scatterlines!(ax,lsH,6*Sz[i,:];linewidth = 2,markersize = 8,
    color = (colors[mod(i,3) + 1],(i-selectedβ[1] + 6)/(selectedβ[end] - selectedβ[1] + 6)),
    label = "$(round(1/lsβ2[i];digits = 2))")
end

insetsize = (width = 120,height = 90)
inset_ax = Axis(fig[1,1];insetsize...,
halign=0.12,    # 水平居中
valign=0.9,    # 垂直底部
# valign=0.34,    # 垂直底部
# halign=0.665,    # 水平居中
backgroundcolor = :white,
xgridvisible=false,    # 关闭网格
ygridvisible=false,
# xticks = ([0.66,0.71],[L"λ_{c1}",L"λ_{c2}"])
xticks = 0:0.5:10,
yticks = 0:0.5:3,
)
for i in selectedβ
    scatterlines!(inset_ax,lsH,6*Sz[i,:];linewidth = 2,markersize = 8,
    color = (colors[mod(i,3) + 1],(i-selectedβ[1] + 6)/(selectedβ[end] - selectedβ[1] + 6)),
    label = "$(round(1/lsβ2[i];digits = 2))")
end
xlims!(inset_ax,1.2,2.2)
ylims!(inset_ax,0.5,1.5)

Legend(fig[1,2],ax,L"T\ /\ J")
# ylims!(0,1/2)
resize_to_layout!(fig)
display(fig)


