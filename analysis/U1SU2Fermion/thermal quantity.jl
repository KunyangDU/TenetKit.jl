using JLD2,CairoMakie,LaTeXStrings,FiniteLattices

include("../analysis/analysis.jl")
include("model.jl")

Lx = 10
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)

params = (μ = 0,)
D = 100

tailname = "_tanTRG"

@load "../codes/examples/U1SU2Hubbard/data/lsβ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ
@load "../codes/examples/U1SU2Hubbard/data/data_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" data

f = data["f"]
u = data["u"]
Ce = data["Ce"]

cβ = easyinterp10(lsβ)

figsize = (height=150,width=300)
fig = Figure()
axf = Axis(fig[1,1];xscale=log10,figsize...,
title = "Spinless free fermion",
ylabel = L"F\ /\ N" )
scatter!(axf, 1 ./ lsβ, f / L)
lines!(axf, 1 ./ easyinterp10(lsβ), fe.(easyinterp10(lsβ),Lx,Ly);color = :red)

axu = Axis(fig[2,1];xscale=log10,figsize...,
ylabel = L"U\ /\ N")
scatter!(axu, 1 ./ lsβ, u / L)
lines!(axu, 1 ./ cβ, ue.(cβ,Lx,Ly);color = :red)

axce = Axis(fig[3,1];xscale=log10,figsize...,
xlabel = L"T",ylabel =L"C_e\ /\ N")
scatter!(axce, 1 ./ lsβ, Ce / L)
lines!(axce, 1 ./ cβ, ce.(cβ,Lx,Ly);color = :red)

hidexdecorations!(axf;ticks = false,grid = false)
hidexdecorations!(axu;ticks = false,grid = false)

resize_to_layout!(fig)
display(fig)

save("U1SU2Fermion/figures/thermal quantity$(tailname).png",fig)

f - fe.(lsβ,Lx,Ly)*size(Latt)
