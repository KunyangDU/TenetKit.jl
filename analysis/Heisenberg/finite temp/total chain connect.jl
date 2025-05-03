using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

SU2name = "../codes/examples/Heisenberg/data/SU2/connect"

dataname = "Heisenberg/data"
D = 2^7
Lx = 14
Ly = 1
@load "$(SU2name)/Latt_$(Lx)x$(Ly).jld2" Latt

trparams = (J=1,)
u1params = (Jz = 1,Jxy = 0.5,h=0)
su2params = (J=1,)
edparams = (Jz=1,Jxy=1)
lsβed = vcat(2. .^ (-5:1:-1), 1:10) .* 2
@load "$(dataname)/data_obc_$(Ly)x$(Lx)_$(edparams).jld2" data
dataed = data
χed = dataed["χ"]
Ced = dataed["C"]
Eed = dataed["E"]


@load "$(SU2name)/lsβ2_$(Lx)x$(Ly)_$(D)_$(trparams).jld2" lsβ2
@load "$(SU2name)/data_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" data
lsβ20 = lsβ2
datasu2 = data
Csu2 = @. real((datasu2["E2"] - datasu2["E"]^2) * lsβ2 ^ 2)
χsu2 = [calcFDT(Latt,datasu2["obs"][i]["SS"]) + 3size(Latt)/4 for i in eachindex(lsβ2)] .* lsβ2
lsEsu2 = datasu2["E"]
extβs = [(1.0,10.0)]
for extβ in extβs
    @load "$(SU2name)/lsβ2_$(Lx)x$(Ly)_$(D)_$(trparams)_connect_$(extβ).jld2" lsβ2
    @load "$(SU2name)/data_$(Lx)x$(Ly)_$(D)_$(su2params)_connect_$(extβ).jld2" data
    cCsu2 = @. real((data["E2"] - data["E"]^2) * lsβ2 ^ 2)
    cχsu2 = [calcFDT(Latt,data["obs"][i]["SS"]) + 3size(Latt)/4 for i in eachindex(lsβ2)] .* lsβ2
    clsEsu2 = data["E"]
    push!(lsβ20,lsβ2...)
    push!(Csu2,cCsu2...)
    push!(χsu2,cχsu2...)
    push!(lsEsu2,clsEsu2...)
end

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
scatter!(axe,1 ./ lsβ20, lsEsu2 / size(Latt),strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU(2)}")

scatter!(axc,1 ./ lsβed,Ced / size(Latt),strokewidth = 2,markersize = 18,strokecolor = :green,marker = :diamond,color = :white,label = L"\mathrm{ED}")
scatter!(axc,1 ./ lsβ20, Csu2 / size(Latt),strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU2}")

scatter!(axχ,1 ./ lsβed,χed / size(Latt),strokewidth = 2,markersize = 18,strokecolor = :green,marker = :diamond,color = :white,label = L"\mathrm{ED}")
scatter!(axχ,1 ./ lsβ20, χsu2 / size(Latt) / 3,strokewidth = 2,markersize = 16,strokecolor = :red,color = :white,label = L"\mathrm{SU2}")

axislegend(axe,position = :lt)

Esu2err = abs.((Eed .- lsEsu2[end-length(lsβed)+1:end]) ./ Eed)
Csu2err = abs.((Ced .- Csu2[end-length(lsβed)+1:end]) ./ Ced)
χsu2err = abs.((χed .- χsu2[end-length(lsβed)+1:end]) ./ χed)

scatterlines!(axeerr,1 ./ lsβed, Esu2err / size(Latt),color = :red,label = L"\mathrm{SU(2)}")
scatterlines!(axcerr,1 ./ lsβed, Csu2err / size(Latt),color = :red,label = L"\mathrm{SU2}")
scatterlines!(axχerr,1 ./ lsβed, χsu2err / size(Latt) / 3,color = :red,label = L"\mathrm{SU2}")

hidexdecorations!(axeerr,grid = false,ticks = false)
hidexdecorations!(axcerr,grid = false,ticks = false)
axislegend(axeerr,position = :cb)
xlims!(axeerr,1/25,10)
xlims!(axcerr,1/25,10)
xlims!(axχerr,1/25,10)

resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/total_$(Lx)x$(Ly)_$(D)_$(trparams)_connect.png",fig)
save("Heisenberg/figures/total_$(Lx)x$(Ly)_$(D)_$(trparams)_connect.pdf",fig)


