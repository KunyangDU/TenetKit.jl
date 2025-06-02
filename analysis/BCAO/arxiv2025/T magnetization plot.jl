
using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

trivialname = "../codes/examples/BCAO/arxiv2025/data/yeesuan"
figurename = "tanTRG/structure factor"
typename = "tanTRG"

gsname = "../codes/examples/BCAO/arxivJ1JzpmJ3/data/yeesuan"


D = 2^8
Dgs = 2^7
Lx = 4
Ly = 4
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt

lsH = vcat(0.1:0.02:0.2)

lsβ2 = let H = lsH[1]
    params1_Kitaev = (J1 = -0.59, K1 = -1, Γ1 = 0.53, Γ1′ = 0.11)
    params23 = (J2 = -0.038, J3xy = 0.31, J3z = 0.0092, Hx = H)
    paramsh = (pinh=0.,)

    params1 = let 
        v = collect(params1_Kitaev)
        v1 = PC2Y*v
        (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
    end

    params = merge(params1,params23,paramsh)
    params_Kitaev = merge(params1_Kitaev,params23,paramsh)
    @load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsβ2
    lsβ2
end

@load "$(trivialname)/Sx_$(Lx)x$(Ly)_$(D).jld2" Sx
@load "$(gsname)/lsSx_$(Lx)x$(Ly)_$(Dgs).jld2" lsSx
@load "$(gsname)/lsHx_$(Lx)x$(Ly)_$(Dgs).jld2" lsHx

figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];
xlabel = L"H\ /\ J",ylabel = L"3\langle\mathbf{S}\cdot\mathbf{\hat{H}}\rangle\ /\ S",
title = "Magnetization, $(Lx)x$(Ly) ZZ-HC-CY, D=$(D)",
# xticks = 0:0.1:1,
xgridvisible=false,    # 关闭网格
ygridvisible=false,
xticks = 0:0.5:10,
yticks = 0:0.5:3,
figsize...)

xlims!(ax,0.5,1.5)
ylims!(ax,0,3)
selectedβ = reverse(length(lsβ2):-4:17)
colors = [:blue,:green,:red]
lines!(ax,[0,5],ones(2);color = :grey,linestyle = :dash)
for i in selectedβ
    scatterlines!(ax,lsH * 5.6,6*Sx[i,:];linewidth = 2,markersize = 8,
    color = (colors[mod(i,3) + 1],(i-selectedβ[1] + 6)/(selectedβ[end] - selectedβ[1] + 6)),
    label = "$(round(1/lsβ2[i];digits = 2))")
end

lines!(ax,lsHx, lsSx * 6, linewidth = 2,label = "DMRG\nD=$(Dgs)")

# insetsize = (width = 120,height = 90)
insetsize = (width = 150,height = 90)

# inset_ax = Axis(fig[1,1];insetsize...,
# # halign=0.12,    # 水平居中
# # valign=0.9,    # 垂直底部
# valign=0.9,    # 垂直底部
# halign=0.145,    # 水平居中
# xgridvisible=false,    # 关闭网格
# ygridvisible=false,
# # xticks = ([0.66,0.71],[L"λ_{c1}",L"λ_{c2}"])
# xticks = 0:0.5:10,
# yticks = 0:0.5:3,
# )
# lines!(inset_ax,[0,5],ones(2);color = :grey,linestyle = :dash)
# for i in selectedβ
#     scatterlines!(inset_ax,lsH,6*Sx[i,:];linewidth = 2,markersize = 8,
#     color = (colors[mod(i,3) + 1],(i-selectedβ[1] + 6)/(selectedβ[end] - selectedβ[1] + 6)),
#     label = "$(round(1/lsβ2[i];digits = 2))")
# end
# xlims!(inset_ax,1.2,2.2)
# ylims!(inset_ax,0.5,1.5)

Legend(fig[1,2],ax,L"T\ /\ J")
# ylims!(0,1/2)
resize_to_layout!(fig)
display(fig)

save("BCAO/arxiv2025/figures/$(typename)/Magnetization Hx_$(Lx)x$(Ly)_$(D).png",fig)
save("BCAO/arxiv2025/figures/$(typename)/Magnetization Hx_$(Lx)x$(Ly)_$(D).pdf",fig)

