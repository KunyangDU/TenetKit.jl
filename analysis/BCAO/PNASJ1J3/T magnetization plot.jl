
using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

trivialname = "../codes/examples/BCAO/PNASJ1J3/data/yeesuan"
figurename = "tanTRG/structure factor"
typename = "tanTRG"

D = 2^8
Lx = 4
Ly = 4
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt

lsH = vcat(0.01:0.002:0.024,0.028:0.004:0.04)

lsβ2 = let H = lsH[1]
    params = (Hx = H, J1xy = -1.0, J1z = -0.158, D = 0.0132, E = -0.0132, J3xy = 0.329, J3z = -0.112)

    @load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
    lsβ2
end

@load "$(trivialname)/Sx_$(Lx)x$(Ly)_$(D).jld2" Sx

figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];
xlabel = L"H\ /\ J",ylabel = L"3\langle\mathbf{S}\cdot\mathbf{\hat{H}}\rangle\ /\ S",
title = "Magnetization, $(Lx)x$(2Ly)x2 ZZ-HC-CY, D=$(D)",
# xticks = 0:0.1:1,
xgridvisible=false,    # 关闭网格
ygridvisible=false,
xticks = 0:0.5:10,
yticks = 0:0.5:3,
figsize...)

xlims!(ax,0.,1.)
ylims!(ax,0,3)
selectedβ = reverse(length(lsβ2):-4:17)
colors = [:blue,:green,:red]


lines!(ax,[0,1],ones(2);color = :grey,linestyle = :dash)
for i in selectedβ
    scatterlines!(ax,lsH * 7.6, 6*Sx[i,:];linewidth = 2,markersize = 8,
    color = (colors[mod(i,3) + 1],(i-selectedβ[1] + 6)/(selectedβ[end] - selectedβ[1] + 6)),
    label = "$(round(1/lsβ2[i];digits = 2))")
end

Legend(fig[1,2],ax,L"T\ /\ J")
# ylims!(0,1/2)
resize_to_layout!(fig)
display(fig)

save("BCAO/PNASJ1J3/figures/$(typename)/Magnetization Hx_$(Lx)x$(Ly)_$(D).png",fig)
save("BCAO/PNASJ1J3/figures/$(typename)/Magnetization Hx_$(Lx)x$(Ly)_$(D).pdf",fig)

