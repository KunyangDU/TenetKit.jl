using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

trivialname = "../codes/examples/Heisenberg/data/trivial"
su2name = "../codes/examples/Heisenberg/data/SU2"
u1name = "../codes/examples/Heisenberg/data/U1"

D = 2^7
# Lx = 4
Ly = 4
trivialparams = (J=1,h=0)
su2params = (J=1,)
u1params = (Jz=1,Jxy=0.5,h=0)

lsLx = 4:2:12

trivialEg = zeros(length(lsLx))
su2Eg = zeros(length(lsLx))
u1Eg = zeros(length(lsLx))

for (i,Lx) in enumerate(lsLx)
    @load "$(trivialname)/lsEg_$(Lx)x$(Ly)_$(D)_$(trivialparams).jld2" lsEg
    lsEgtrivial = lsEg
    @load "$(su2name)/lsEg_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" lsEg
    lsEgsu2 = lsEg
    @load "$(u1name)/lsEg_$(Lx)x$(Ly)_$(D)_$(u1params).jld2" lsEg
    lsEgu1 = lsEg

    trivialEg[i] = lsEgtrivial[end] / Lx / Ly - 1/4
    su2Eg[i] = lsEgsu2[end] / Lx / Ly - 1/4
    u1Eg[i] = lsEgu1[end] / Lx / Ly - 1/4
end
figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xlabel = L"L_x",ylabel = L"(E-E_A)/N",
title = "Heisenberg, SquaCY, D = $(D)",
xticks = lsLx)
scatter!(ax,lsLx,su2Eg,strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU(2)}")
scatterlines!(ax,lsLx,trivialEg,markersize = 13,label = L"\mathrm{NonSym.}")
scatter!(ax,lsLx,u1Eg,markersize = 10,marker = :star4,color = :gold,label = L"\mathrm{U(1)}")
axislegend(ax,position = :rt)
resize_to_layout!(fig)
display(fig)
save("Heisenberg/figures/compare Eg_$(Lx)x$(Ly)_$(D).png",fig)
save("Heisenberg/figures/compare Eg_$(Lx)x$(Ly)_$(D).pdf",fig)


