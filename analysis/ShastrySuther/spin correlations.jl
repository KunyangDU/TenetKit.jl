

using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../analysis/analysis.jl")
include("model.jl")
Ly = 1
tailname = "SU2"
totalname = "examples/ShastrySuther/data"

lsλ = vcat(0:0.05:0.6,0.61:0.01:0.8,0.85:0.05:1)
Lx = 4
Ly = 4
@load "../codes/$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9
J1 = 1

@load "../codes/$(totalname)/dimerOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" dimerOP
@load "../codes/$(totalname)/crossOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" crossOP
@load "../codes/$(totalname)/nullOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nullOP
@load "../codes/$(totalname)/plaquetteOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" plaquetteOP
@load "../codes/$(totalname)/NNSSOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" NNSSOP
@load "../codes/$(totalname)/AFMOD_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" AFMOD

figsize = (width = 400,height = 300)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "Spin correlations of Shastry-Suther model\n $(Lx)x$(Ly),  D = $(D),  $(tailname)",
ylabel = L"\langle \mathbf{S}_i\cdot \mathbf{S}_j\rangle",
xlabel = L"J_1/J_2",
xticks = 0:0.1:1,
yticks = -2:0.1:2
)
# axnnn = Axis(fig[2,1];figsize...,
# ylabel = L"{\mathrm{NNN}}",
# xlabel = L"J_1/J_2",
# xticks = 0:0.1:1,
# # yticks = -2:0.1:2
# )

# for (ax,data) in [(axnn,NNSSOP),(axnnn,vcat(dimerOP,crossOP,nullOP))]



#     ylims!(ax,[-0.01,0.2] .+ collect(extrema(data))...)
# end
data = vcat(NNSSOP,dimerOP,crossOP,nullOP)

polyps(x,y) = Point2f[ (x[1], y[1]),  (x[2], y[1]), (x[2], y[2]), (x[1], y[2])]
poly_dimer = polyps((0,0.66),extrema(data))
poly_SL = polyps((0.66,0.71),extrema(data))
poly_Neel = polyps((0.71,1),extrema(data))
poly!(ax, poly_dimer; color=(:goldenrod,0.2))
poly!(ax, poly_SL; color=(:blue,0.2))
poly!(ax, poly_Neel; color=(:green,0.2))

scatterlines!(ax,lsλ,NNSSOP,linewidth = 2,markersize = 10,label = L"{\mathrm{NN}}")
scatterlines!(ax,lsλ,dimerOP,label = L"J_2\ {\mathrm{NNN}}",linewidth = 2,markersize = 10)
scatterlines!(ax,lsλ,crossOP,label = L"\mathrm{Cross}\ {\mathrm{NNN}}",linewidth = 2,markersize = 10)
scatterlines!(ax,lsλ,nullOP,label = L"\mathrm{Null}\ {\mathrm{NNN}}",linewidth = 2,markersize = 10)

text!(ax,0.3,0.2;text = L"\mathrm{Dimer}",fontsize = 18,color = :goldenrod)
text!(ax,0.64,0.2;text = L"\mathbb{Z}_2\ \mathrm{SL}",fontsize = 18,color = :blue)
text!(ax,0.82,0.2;text = L"\mathrm{Neel}",fontsize = 18,color = :green)

xlims!(ax,0,1)
ylims!(ax,-0.8,0.3)

insetsize = (width = 200,height = 120)
inset_ax = Axis(fig[1,1];insetsize...,
halign=0.25,    # 水平居中
valign=0.39,    # 垂直底部
backgroundcolor = :white,
xgridvisible=false,    # 关闭网格
ygridvisible=false,
xticks = ([0.66,0.71],[L"λ_{c1}",L"λ_{c2}"])
)

lines!(inset_ax,0.66 * ones(2),[-1,1];color = :grey)
lines!(inset_ax,0.71 * ones(2),[-1,1];color = :grey)
scatterlines!(inset_ax,lsλ,NNSSOP,linewidth = 2,markersize = 10,label = L"{\mathrm{NN}}")
scatterlines!(inset_ax,lsλ,dimerOP,label = L"J_2\ {\mathrm{NNN}}",linewidth = 2,markersize = 10)
scatterlines!(inset_ax,lsλ,crossOP,label = L"\mathrm{Cross}\ {\mathrm{NNN}}",linewidth = 2,markersize = 10)
scatterlines!(inset_ax,lsλ,nullOP,label = L"\mathrm{Null}\ {\mathrm{NNN}}",linewidth = 2,markersize = 10)
xlims!(inset_ax,0.6,0.8)
ylims!(inset_ax,-0.75,0.25)

axislegend(ax,position = :rb)
resize_to_layout!(fig)
display(fig)

save("ShastrySuther/figures/spin correlations_$(Lx)x$(Ly)_D=$(D)_$(tailname).png",fig)
save("ShastrySuther/figures/spin correlations_$(Lx)x$(Ly)_D=$(D)_$(tailname).pdf",fig)