using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("model.jl")

typename = "trivial"
dataname = "../codes/examples/Heisenberg/data/triangle/$(typename)"
D = 100
Lx = 6
Ly = 6

lsH = 0:0.2:5
Sz = zeros(length(lsH))
for (i,H) in enumerate(lsH)
params = (J=1,H=H)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
Sz[i] = sum([gsdata["Sz"][(i,)] for i in 1:size(Latt)]) / size(Latt)
end

figsize = (width = 400,height = 200)
fig = Figure()

ax = Axis(fig[1,1];figsize...,
xlabel = L"H\ /\ J",ylabel = L"3\langle\mathbf{S}\cdot\mathbf{\hat{H}}\rangle\ /\ S",
title = "Magnetization, $(Lx)x$(Ly) ZZ-HC-CY, D=$(D)",
# xticks = 0:0.1:1,
xticks = 0:0.5:10,
yticks = 0:0.5:3,
)
scatterlines!(ax,lsH,Sz*2*3)
# xlims!(ax,0,5.0)
# ylims!(ax,0,0.5)

# insetsize = (width = 160,height = 90)
# inset_ax = Axis(fig[1,1];insetsize...,
# halign=0.83,    # 水平居中
# # valign=0.34,    # 垂直底部
# # halign=0.665,    # 水平居中
# valign=0.2,    # 垂直底部
# backgroundcolor = :white,
# xgridvisible=false,    # 关闭网格
# ygridvisible=false,
# # xticks = ([0.66,0.71],[L"λ_{c1}",L"λ_{c2}"])
# xticks = 0:0.5:1.5,yticks = 0:0.15:0.3
# )
# scatterlines!(inset_ax,lsHx * 6.54,lsSx)
# xlims!(inset_ax,0,1.5)
# ylims!(inset_ax,0,0.3)

resize_to_layout!(fig)
display(fig)

save("Heisenberg/triangle/figures/$(typename)/Magnetization Hx_$(Lx)x$(Ly)_$(D).png",fig)
save("Heisenberg/triangle/figures/$(typename)/Magnetization Hx_$(Lx)x$(Ly)_$(D).pdf",fig)


