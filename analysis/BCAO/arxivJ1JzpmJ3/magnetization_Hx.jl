using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/BCAO/arxivJ1JzpmJ3/data/yeesuan"
figurename = "BCAO/arxivJ1JzpmJ3/figures"

D = 2^8
Lx = 6
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

select_point = div(size(Latt),Ly) |> x -> x+1:size(Latt)-x
# select_point = 1:size(Latt)
lsHx = vcat(0.0:0.02:0.2,0.25:0.05:0.5)
lsSx = zeros(length(lsHx))
for (i,Hx) in enumerate(lsHx)
    params = (Hx = Hx, J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
    lsSx[i] = [gsdata["Sx"][(i,)] for i in select_point] |> x -> sum(x) / length(x)
end

lsHx *= 6.54

@save "$(dataname)/lsSx_$(Lx)x$(Ly)_$(D).jld2" lsSx
@save "$(dataname)/lsHx_$(Lx)x$(Ly)_$(D).jld2" lsHx

figsize = (width = 400,height = 200)
figsizeinset = (width = 200,height = 100)

fig = Figure()

ax = Axis(fig[1,1];figsize...,
xlabel = L"\mu_0H\ /\ \mathrm{T}",ylabel = L"\mathbf{S}\cdot\mathbf{\hat{H}}",
title = "Magnetization, $(Ly)x$(Lx)x2 ZZ-HC-CY, D=$(D)",
# xticks = 0:0.1:1,
xticks = 0:0.5:10,
yticks = 0:0.05:0.5,
xgridvisible=false,    # 关闭网格
ygridvisible=false,
)
# xlims!(ax,extrema(lsHx))
lines!(ax,lsHx,lsSx)
xlims!(ax,0,5.0)
ylims!(ax,0,0.45)

insetsize = (width = 160,height = 100)
inset_ax = Axis(fig[1,1];insetsize...,
halign=0.83,    # 水平居中
# valign=0.34,    # 垂直底部
# halign=0.665,    # 水平居中
valign=0.3,    # 垂直底部
backgroundcolor = :white,
# xticks = ([0.66,0.71],[L"λ_{c1}",L"λ_{c2}"])
xticks = 0:0.5:1.5,yticks = ([0,1/6,1/3],[L"0",L"1/3",L"2/3"]),
)
scatterlines!(inset_ax,lsHx,lsSx)
xlims!(inset_ax,0,1.5)
ylims!(inset_ax,0,1/3)
# scatter!(ax,0.2*6.54,0)
resize_to_layout!(fig)
display(fig)

save("$(figurename)/Magnetization Hx_$(Lx)x$(Ly)_$(D)_$(length(lsHx)).png",fig)
save("$(figurename)/Magnetization Hx_$(Lx)x$(Ly)_$(D)_$(length(lsHx)).pdf",fig)


