using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../model.jl")
Ly = 1
tailname = "SU2"
totalname = "examples/J1J2chain/data/rescale"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
# lsλ = 0:0.1:1
# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1,1.2:0.2,2)
lsλp = vcat(0.5:0.5:5)
# lsλp = vcat(0.2:0.2:2,2.5:0.5:5)
lsλ = vcat(-reverse(lsλp),0,lsλp)
lsLx = [20,]
Lx = 20
@load "../codes/$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9
J1 = 1

@load "J1J2chain/data/rescale/NNAFM_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nnafm
@load "J1J2chain/data/rescale/NNNAFM_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nnnafm
@load "J1J2chain/data/rescale/DOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nndop
@load "J1J2chain/data/rescale/NNcoefficient_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nncoeff
@load "J1J2chain/data/rescale/NNNcoefficient_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nnncoeff
@load "J1J2chain/data/rescale/NNFM_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nnfm
@load "J1J2chain/data/rescale/MAXSS_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" maxss

figsize = (width = 600,height = 300)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "Order parameter",
ylabel = L"values",
xticks = -5:0.5:5,
xlabel = L"J_1/J_2")

scatterlines!(ax,lsλ,maxss,label = L"\mathrm{max}\ F(k)",color = :grey,linewidth = 2,markersize = 10)
scatterlines!(ax,lsλ,nnafm,label = L"F(\pi)",color = :green,linewidth = 2,markersize = 10)
scatterlines!(ax,nncoeff[1,:],nncoeff[2,:],label = L"K_{NN}",color = :green,marker = :star5,linewidth = 2,markersize = 10)
scatterlines!(ax,lsλ,nndop,label = L"DOP",color = :purple,linewidth = 2,markersize = 10)
scatterlines!(ax,lsλ,nnnafm,label = L"F(\pi/2)",color = :blue,linewidth = 2,markersize = 10)
scatterlines!(ax,nnncoeff[1,:],nnncoeff[2,:],label = L"K_{NNN}",color = :blue,marker = :star5,linewidth = 2,markersize = 10)
scatterlines!(ax,lsλ,nnfm,label = L"F(0)",color = :red,linewidth = 2,markersize = 10)

text!(ax,-5,2.5;text = "NN FM",color = :red)
text!(ax,-3.5,2.5;text = "SDW",color = :grey)
text!(ax,-1,2.5;text = "NNN AFM",color = :blue)
text!(ax,1.5,2.5;text = "Dimer",color = :purple)
text!(ax,3.5,2.5;text = "NN AFM",color = :green)

lines!(ax,-4 * ones(2),[0,3],color = :grey,linestyle = :dash)
lines!(ax,-2 * ones(2),[0,3],color = :grey,linestyle = :dash)
lines!(ax,1 * ones(2),[0,3],color = :grey,linestyle = :dash)
lines!(ax,3 * ones(2),[0,3],color = :grey,linestyle = :dash)

Legend(fig[1,2],ax)

resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/rescale/phase diagram_$(Lx)x$(Ly)_D=$(D)_$(tailname).png",fig)
save("J1J2chain/figures/rescale/phase diagram_$(Lx)x$(Ly)_D=$(D)_$(tailname).pdf",fig)

