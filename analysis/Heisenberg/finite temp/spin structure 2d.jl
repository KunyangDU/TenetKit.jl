using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/data/trivial"

Lx = 8
Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 256
DS = 2^4
τ = 0.5
Nhot = -20
βmax = 10

params = (J = 1.0, Δ = 1.0, Hz = 0.0)

edgepoints = vcat(1:2Ly, size(Latt)-2Ly:size(Latt))
# edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
# lsβ2 = vcat(lsβ2,42:2:200)
validinds = [8,18,20,38]


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

lsax = Axis[]

figsize = (width = 200, height = 200)
pos = [(1,1),(1,2),(2,1),(2,2)]
fig = Figure()
figsize = (width = 200, height = 200)
Sm = 5
for (ii,i) in enumerate(validinds)
    ax = Axis(fig[pos[ii]...];figsize...,
    title = "β = $(round(lsβ2[i];digits = 2))")
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    FSxSx,FSySy,FSzSz = map(x -> FT2(getCorrMat(Latt,data["obs"][x],1/4;selected_point = 1:size(Latt)),Latt,lstk),[("Sx","Sx"),("Sy","Sy"),("Sz","Sz")])
    hm = heatmap!(ax,x / pi,y / pi,FSxSx .+ FSySy .+ FSzSz,colorrange = (0,Sm))
    push!(lsax,ax)
end

hidexdecorations!(lsax[1],ticks = false,minorticks = false)
hidexdecorations!(lsax[2],ticks = false,minorticks = false)
hideydecorations!(lsax[2],ticks = false,minorticks = false)
hideydecorations!(lsax[4],ticks = false,minorticks = false)

Colorbar(fig[1,3],label = L"S(k)",colorrange = (0,Sm))
Colorbar(fig[2,3],label = L"S(k)",colorrange = (0,Sm))

resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/spin_structure_2d_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/figures/spin_structure_2d_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
