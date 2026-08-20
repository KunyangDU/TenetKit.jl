using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/data/trivial"

D = 128
Lx = 8
Ly = 6
params = (J=1, Δ =1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# @load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

# Sz = real.([gsdata[("Sz",)][(i,)] for i in 1:size(Latt)])

# Sz = [gsdata["Sz"][(i,)] for i in 1:size(Latt)]


fig = Figure()
ax = Axis(fig[1,1])


lskx = pi * range(0,1,75)
lsky = pi * range(0,1,75)
lsk = [[kx,ky] for kx in lskx for ky in lsky]
lstk = map(x -> Tuple(x),lsk)
x = map(lsk) do k
    k[1]
end
y = map(lsk) do k
    k[2]
end

FSxSx,FSySy,FSzSz = map(x -> FT2(getCorrMat(Latt,gsdata[x],1/4;selected_point = 1:size(Latt)),Latt,lstk),[("Sx","Sx"),("Sy","Sy"),("Sz","Sz")])


Sm = maximum(FSxSx .+ FSySy .+ FSzSz)
Smd = maximum(vcat(FSxSx,FSySy,FSzSz))*1.
figsize = (width = 200,height = 200)

fig = Figure()
ax = Axis(fig[1,1];autolimitaspect = true,figsize...,
xlabel = L"k_x\ /\ \pi",ylabel = L"k_y\ /\ \pi",
title = L"\langle \mathbf{S}\cdot\mathbf{S}\rangle")
# xticks = (4pi/sqrt(3)*(-1:0.5:1),string.(-2:2)),yticks = (2pi*(-1:0.5:1),string.(-2:1:2)),)
hm = heatmap!(ax,x / pi,y / pi,FSxSx .+ FSySy .+ FSzSz,colorrange = (0,Sm))
Colorbar(fig[1,2],hm;label = L"S(k)")
resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/trivial_spin structure_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
save("Heisenberg/figures/trivial_spin structure_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)

# gsdata