using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

trivialname = "../codes/examples/Heisenberg/data/trivial"
SU2name = "../codes/examples/Heisenberg/data/SU2"
U1name = "../codes/examples/Heisenberg/data/U1"
dataname = "Heisenberg/data"
D = 2^7
Lx = 14
Ly = 1
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt

trparams = (J=1,h=0)
u1params = (Jz = 1,Jxy = 0.5,h=0)
su2params = (J=1,)
edparams = (Jz=1,Jxy=1)
lsβed = vcat(2. .^ (-5:1:-1), 1:10) .* 2
@load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(trparams).jld2" lsβ2

@load "$(trivialname)/lsE_$(Lx)x$(Ly)_$(D)_$(trparams).jld2" lsE
lsEtr = real.(lsE)
@load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(trparams).jld2" data
datatr = data
Ctr = @. real((datatr["E2"] - datatr["E"]^2) * lsβ2 ^ 2)
χtr = @. real((datatr["Mz2"] - datatr["Mz"]^2) * lsβ2)

@load "$(U1name)/lsE_$(Lx)x$(Ly)_$(D)_$(u1params).jld2" lsE
lsEu1 = real.(lsE)
@load "$(U1name)/data_$(Lx)x$(Ly)_$(D)_$(u1params).jld2" data
datau1 = data
Cu1 = @. real((datau1["E2"] - datau1["E"]^2) * lsβ2 ^ 2)
χu1 = @. real((datau1["Mz2"] - datau1["Mz"]^2) * lsβ2)

@load "$(SU2name)/lsE_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" lsE
lsEsu2 = real.(lsE)
@load "$(SU2name)/data_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" data
datasu2 = data
Csu2 = @. real((datasu2["E2"] - datasu2["E"]^2) * lsβ2 ^ 2)
χsu2 = @. real((datasu2["M2"]) * lsβ2)

@load "$(dataname)/data_obc_$(Ly)x$(Lx)_$(edparams).jld2" data
dataed = data
χed = dataed["χ"]
Ced = dataed["C"]
Eed = dataed["E"]


figsize = (width = 300,height = 150)

fig = Figure()
axe = Axis(fig[1,1];figsize...,
title = "Heisenberg chain $(Lx)x$(Ly), D = $(D)",titlealign = :left,
xscale = log10,
xlabel = L"T\cdot J",ylabel=L"E/N")
axc = Axis(fig[2,1];figsize...,
xscale = log10,
xlabel = L"T\cdot J",ylabel=L"C/N")
axχ = Axis(fig[3,1];figsize...,
xscale = log10,
xlabel = L"T\cdot J",ylabel=L"\chi/N")

axeerr = Axis(fig[1,2];figsize...,
xscale = log10,yscale = log10,
xlabel = L"T\cdot J",ylabel=L"\mathrm{Err}\ \left(E/N\right)")
axcerr = Axis(fig[2,2];figsize...,
xscale = log10,yscale = log10,
xlabel = L"T\cdot J",ylabel=L"\mathrm{Err}\ \left(C/N\right)")
axχerr = Axis(fig[3,2];figsize...,
xscale = log10,yscale = log10,
xlabel = L"T\cdot J",ylabel=L"\mathrm{Err}\ \left(\chi/N\right)")

xlims!(axe,1/25,10)
xlims!(axc,1/25,10)
# ylims!(axc,-0.1,1)
xlims!(axχ,1/25,10)
# ylims!(axχ,-0.1,1)

hidexdecorations!(axe,grid = false,ticks = false)
hidexdecorations!(axc,grid = false,ticks = false)

scatter!(axe,1 ./ lsβed, Eed / size(Latt),strokewidth = 2,markersize = 18,strokecolor = :green,marker = :diamond,color = :white,label = L"\mathrm{ED}")
scatter!(axe,1 ./ lsβ2, lsEsu2 / size(Latt),strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU(2)}")
scatterlines!(axe,1 ./ lsβ2, lsEu1 / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{U(1)}")
scatter!(axe,1 ./ lsβ2, lsEtr / size(Latt),markersize = 8,marker = :star4,color = :gold,label = L"\mathrm{NonSym.}")

scatter!(axc,1 ./ lsβed,Ced / size(Latt),strokewidth = 2,markersize = 18,strokecolor = :green,marker = :diamond,color = :white,label = L"\mathrm{ED}")
scatter!(axc,1 ./ lsβ2, Csu2 / size(Latt),strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU2}")
scatterlines!(axc,1 ./ lsβ2, Cu1 / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{U(1)}")
scatter!(axc,1 ./ lsβ2, Ctr / size(Latt),markersize = 8,marker = :star4,color = :gold,label = L"\mathrm{NonSym.}")

scatter!(axχ,1 ./ lsβed,χed / size(Latt),strokewidth = 2,markersize = 18,strokecolor = :green,marker = :diamond,color = :white,label = L"\mathrm{ED}")
scatter!(axχ,1 ./ lsβ2, χsu2 / size(Latt) / 3,strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU2}")
scatterlines!(axχ,1 ./ lsβ2, χu1 / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{U(1)}")
scatter!(axχ,1 ./ lsβ2, χtr / size(Latt),markersize = 8,marker = :star4,color = :gold,label = L"\mathrm{NonSym.}")

axislegend(axe,position = :lt)

Esu2err = abs.((Eed .- lsEsu2[end-length(lsβed)+1:end]) ./ Eed)
Eu1err = abs.((Eed .- lsEu1[end-length(lsβed)+1:end]) ./ Eed)
Etrerr = abs.((Eed .- lsEtr[end-length(lsβed)+1:end]) ./ Eed)
Csu2err = abs.((Ced .- Csu2[end-length(lsβed)+1:end]) ./ Ced)
Cu1err = abs.((Ced .- Cu1[end-length(lsβed)+1:end]) ./ Ced)
Ctrerr = abs.((Ced .- Ctr[end-length(lsβed)+1:end]) ./ Ced)
χsu2err = abs.((χed .- χsu2[end-length(lsβed)+1:end]) ./ χed)
χu1err = abs.((χed .- χu1[end-length(lsβed)+1:end]) ./ χed)
χtrerr = abs.((χed .- χtr[end-length(lsβed)+1:end]) ./ χed)

scatterlines!(axeerr,1 ./ lsβed, Esu2err / size(Latt),color = :red,label = L"\mathrm{SU(2)}")
scatterlines!(axeerr,1 ./ lsβed, Eu1err / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{U(1)}")
scatterlines!(axeerr,1 ./ lsβed, Etrerr / size(Latt),color = :gold,label = L"\mathrm{NonSym.}")

scatterlines!(axcerr,1 ./ lsβed, Csu2err / size(Latt),color = :red,label = L"\mathrm{SU2}")
scatterlines!(axcerr,1 ./ lsβed, Cu1err / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{U(1)}")
scatterlines!(axcerr,1 ./ lsβed, Ctrerr / size(Latt),color = :gold,label = L"\mathrm{NonSym.}")

scatterlines!(axχerr,1 ./ lsβed, χsu2err / size(Latt) / 3,color = :red,label = L"\mathrm{SU2}")
scatterlines!(axχerr,1 ./ lsβed, χu1err / size(Latt),linewidth = 2,markersize = 12,label = L"\mathrm{U(1)}")
scatterlines!(axχerr,1 ./ lsβed, χtrerr / size(Latt),color = :gold,label = L"\mathrm{NonSym.}")

hidexdecorations!(axeerr,grid = false,ticks = false)
hidexdecorations!(axcerr,grid = false,ticks = false)
axislegend(axeerr,position = :rt)
xlims!(axeerr,1/25,10)
xlims!(axcerr,1/25,10)
xlims!(axχerr,1/25,10)
resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/total_$(Lx)x$(Ly)_$(D)_$(trparams).png",fig)
save("Heisenberg/figures/total_$(Lx)x$(Ly)_$(D)_$(trparams).pdf",fig)



