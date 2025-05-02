using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

trivialname = "../codes/examples/Heisenberg/data/trivial"
U1name = "../codes/examples/Heisenberg/data/U1"
SU2name = "../codes/examples/Heisenberg/data/SU2"

D = 2^8
Lx = 4
Ly = 4
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt

trparams = (J=1,h=0)
u1params = (Jz = 1,Jxy = 0.5,h=0)
su2params = (J=1,)
edparams = (Jz=1,Jxy=1)
lsβed = vcat(2. .^ (-5:1:-1), 1:10) .* 2
@load "$(SU2name)/lsβ2_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" lsβ2
lsβ2u1 = lsβ2
@load "$(trivialname)/lsE_$(Lx)x$(Ly)_$(D)_$(trparams).jld2" lsE
lsEtr = real.(lsE)
@load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(trparams).jld2" data
datatr = data
Ctr = @. real((datatr["E2"] - datatr["E"]^2) * lsβ2 ^ 2)
χtr = @. real((datatr["Mz2"] - datatr["Mz"]^2) * lsβ2)

@load "$(SU2name)/lsE_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" lsE
lsEsu2 = real.(lsE)
@load "$(SU2name)/data_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" data
datasu2 = data
Csu2 = @. real((datasu2["E2"] - datasu2["E"]^2) * lsβ2 ^ 2)
χsu2 = @. real((datasu2["M2"]) * lsβ2)
# D = 2^10
@load "$(U1name)/lsE_$(Lx)x$(Ly)_$(D)_$(u1params).jld2" lsE
lsEu1 = real.(lsE)
@load "$(U1name)/data_$(Lx)x$(Ly)_$(D)_$(u1params).jld2" data
datau1 = data
Cu1 = (@. real((datau1["E2"] - datau1["E"]^2) * lsβ2 ^ 2))
χu1 = (@. real((datau1["Mz2"] - datau1["Mz"]^2) * lsβ2))


figsize = (width = 400,height = 150)

fig = Figure()
axe = Axis(fig[1,1];figsize...,
title = "Heisenberg SquaYC $(Lx)x$(Ly), D = $(D)",titlealign = :left,
xscale = log10,
xlabel = L"T\cdot J",ylabel=L"E/N")
axc = Axis(fig[2,1];figsize...,
xscale = log10,
xlabel = L"T\cdot J",ylabel=L"C/N")
axχ = Axis(fig[3,1];figsize...,
xscale = log10,
xlabel = L"T\cdot J",ylabel=L"\chi/N")

xlims!(axe,1/25,10)
xlims!(axc,1/25,10)
# ylims!(axc,-0.1,1)
xlims!(axχ,1/25,10)
# ylims!(axχ,-0.1,1)

hidexdecorations!(axe,grid = false,ticks = false)
hidexdecorations!(axc,grid = false,ticks = false)

scatter!(axe,1 ./ lsβ2, lsEsu2 / size(Latt),strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU(2)}")
scatterlines!(axe,1 ./ lsβ2, lsEtr / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{NonSym.}")
scatter!(axe,1 ./ lsβ2, lsEu1 / size(Latt),markersize = 8,marker = :star4,color = :gold,label = L"\mathrm{U(1)}")

scatter!(axc,1 ./ lsβ2, Csu2 / size(Latt),strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU2}")
scatterlines!(axc,1 ./ lsβ2, Ctr / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{NonSym.}")
scatter!(axc,1 ./ lsβ2u1, Cu1 / size(Latt),markersize = 8,marker = :star4,color = :gold,label = L"\mathrm{U(1)}")

scatter!(axχ,1 ./ lsβ2, χsu2 / size(Latt) / 3,strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU2}")
scatterlines!(axχ,1 ./ lsβ2, χtr / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{NonSym.}")
scatter!(axχ,1 ./ lsβ2u1, χu1 / size(Latt),markersize = 8,marker = :star4,color = :gold,label = L"\mathrm{U(1)}")

axislegend(axe,position = :rb)

resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/total_$(Lx)x$(Ly)_$(D)_$(trparams).png",fig)
save("Heisenberg/figures/total_$(Lx)x$(Ly)_$(D)_$(trparams).pdf",fig)

# lsEu1 - lsEsu2
# lsEtr - lsEsu2