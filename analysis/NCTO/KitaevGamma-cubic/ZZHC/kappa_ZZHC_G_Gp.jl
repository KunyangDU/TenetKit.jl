using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/ZZHC"
figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC"
tailname = ""

D = 128
Lx = 3
Ly = 4

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt



lsHf = 0:0.02:0.8


θ = 0.0 * pi
ϕ = 0.5 * pi


J = -0.1
K = -1.0
Γ = 0.3
Γ′ = -0.02

params1_Kitaev = (J = J, K = K, Γ = Γ, Γ′ = Γ′)

@load "$(dataname)/lsHf_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).jld2" lsHf
@load "$(dataname)/lsκ_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).jld2" lsκ


figsize = (width = 300,height = 150)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xlabel = L"h",ylabel = L"\chi_M",
title = "$(Ly)x$(Lx), D=$(D), χ_M")
scatterlines!(ax,lsHf[1:end-1],lsκ)
# scatterlines!(ax,lsHf,lsM)
xlims!(ax,extrema(lsHf))
resize_to_layout!(fig)
display(fig)
# lsχ

save("$(figurename)/κ_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).png",fig)
save("$(figurename)/κ_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).pdf",fig)
