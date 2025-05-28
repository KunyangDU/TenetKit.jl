using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/BCAO/PNASJ1J3/data/yeesuan"
figurename = "BCAO/PNASJ1J3/figures"

D = 2^8
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

select_point = div(size(Latt),Ly) |> x -> x+1:size(Latt)-x
# select_point = 1:size(Latt)
# lsHx = vcat(0.0:0.0025:0.1,0.11:0.01,0.2,0.3:0.1:1)
lsHx = vcat(0:0.005:0.095,0.1:0.1:0.5,1.0)
lsSx = zeros(length(lsHx))
for (i,Hx) in enumerate(lsHx)
    params = (Hx = Hx, J1xy = -1.0, J1z = -0.158, D = 0.0132, E = -0.0132, J3xy = 0.329, J3z = -0.112)
    # params = (Hx = Hx, J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
    lsSx[i] = [gsdata["Sx"][(i,)] for i in select_point] |> x -> sum(x) / length(x)
end
figsize = (width = 350,height = 200)
figsizeinset = (width = 200,height = 100)

fig = Figure()

ax = Axis(fig[1,1];figsize...,
xlabel = L"\mu_0H\ /\ \mathrm{T}",ylabel = L"\mathbf{S}\cdot\mathbf{\hat{H}}",
title = "Magnetization, $(Ly)x$(2Lx)x2 ZZ-HC-CY, D=$(D)",
# xticks = 0:0.1:1,
xgridvisible=false,    # 关闭网格
ygridvisible=false,
xticks = 0:0.5:10,
yticks = 0:0.05:0.5,
)
# xlims!(ax,extrema(lsHx))
lines!(ax,lsHx * 7.6,lsSx,linewidth = 2)

lines!(ax,[0,0.5],ones(2)/3;color = :black,linewidth = 1)
lines!(ax,ones(2)*0.5,[0,1/3];color = :black,linewidth = 1)

xlims!(ax,0,5.0)
ylims!(ax,0,0.45)

insetsize = (width = 200,height = 135)
inset_ax = Axis(fig[1,1];insetsize...,
halign=0.9,    # 水平居中
valign=0.37,    # 垂直底部
backgroundcolor = :white,
xticks = 0:0.1:1.5,yticks = ([0,1/6,1/3],[L"0",L"1/3",L"2/3"])
)

lines!(inset_ax,collect(extrema(lsHx)) * 7.6,ones(2)/6, color = :grey, linestyle = :dash)

scatterlines!(inset_ax,lsHx * 7.6,lsSx,linewidth = 2)
xlims!(inset_ax,0,0.5)
ylims!(inset_ax,0,1/3)
# scatter!(ax,0.2*7.6,0)



resize_to_layout!(fig)
display(fig)

save("$(figurename)/Magnetization Hx_$(Lx)x$(Ly)_$(D)_$(length(lsHx)).png",fig)
save("$(figurename)/Magnetization Hx_$(Lx)x$(Ly)_$(D)_$(length(lsHx)).pdf",fig)


