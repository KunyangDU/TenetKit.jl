using JLD2,CairoMakie,LaTeXStrings

include("../analysis/analysis.jl")
include("model.jl")

Lx = 6
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)

params = (μ = 0,)
D = 60

tailname = ""
dataname = "examples/TrivialSpinlessFermion/data"
@load "../codes/$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ2
@load "../codes/$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" data

f = data["f"]
u = data["u"]
Ce = data["Ce"]

cβ = easyinterp10(lsβ2)

# figsize = (height=150,width=300)
# fig = Figure()
# axf = Axis(fig[1,1];xscale=log10,figsize...,
# title = "Spinless free fermion",
# ylabel = L"F\ /\ N" )
# scatter!(axf, 1 ./ lsβ2, f / L)
# lines!(axf, 1 ./ easyinterp10(lsβ2), fe.(easyinterp10(lsβ2),Lx,Ly);color = :red)

# axu = Axis(fig[2,1];xscale=log10,figsize...,
# ylabel = L"U\ /\ N")
# scatter!(axu, 1 ./ lsβ2, u / L)
# lines!(axu, 1 ./ cβ, ue.(cβ,Lx,Ly);color = :red)

# axce = Axis(fig[3,1];xscale=log10,figsize...,
# xlabel = L"T",ylabel =L"C_e\ /\ N")
# scatter!(axce, 1 ./ lsβ2, Ce / L)
# lines!(axce, 1 ./ cβ, ce.(cβ,Lx,Ly);color = :red)

# hidexdecorations!(axf;ticks = false,grid = false)
# hidexdecorations!(axu;ticks = false,grid = false)

# resize_to_layout!(fig)
# display(fig)

# save("TrivialSpinlessFermion/figures/thermal quantity$(tailname).png",fig)

f - fe.(lsβ2,Lx,Ly)*size(Latt)
