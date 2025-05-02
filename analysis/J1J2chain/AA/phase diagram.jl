using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")
Ly = 1
tailname = "SU2"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
# lsλ = 0:0.1:1
# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1,1.2:0.2,2)
lsλ = 0:0.1:2
lsLx = [20,]
Lx = 20
@load "../codes/examples/J1J2chain/data/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9
J1 = 1

@load "J1J2chain/data/NNAFM_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nnafm
@load "J1J2chain/data/NNNAFM_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nnnafm
@load "J1J2chain/data/DOP_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nndop
@load "J1J2chain/data/NNcoefficient_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nncoeff
@load "J1J2chain/data/NNNcoefficient_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nnncoeff

figsize = (width = 500,height = 300)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "Order parameter",
ylabel = L"values",
xlabel = L"J_2/J_1")
scatterlines!(ax,lsλ,nnafm,label = L"F(\pi)",color = :green,linewidth = 2,markersize = 10)
scatterlines!(ax,nncoeff[1,:],nncoeff[2,:],label = L"K_{NN}",color = :green,marker = :star5,linewidth = 2,markersize = 10)
scatterlines!(ax,lsλ,nndop,label = L"DOP",color = :purple,linewidth = 2,markersize = 10)
scatterlines!(ax,lsλ,nnnafm,label = L"F(\pi/2)",color = :blue,linewidth = 2,markersize = 10)
scatterlines!(ax,nnncoeff[1,:],nnncoeff[2,:],label = L"K_{NNN}",color = :blue,marker = :star5,linewidth = 2,markersize = 10)

Legend(fig[1,2],ax)

resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/phase diagram_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).png",fig)
save("J1J2chain/figures/phase diagram_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).pdf",fig)

