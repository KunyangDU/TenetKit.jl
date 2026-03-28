using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("../model.jl")

v(Δ::Number,J::Number = 1, tol::Number = 1e-8) = abs(Δ-1) < tol ? pi*J/2 : pi*J*sqrt(1 - Δ^2)/acos(Δ)/2


trivialname = "../codes/examples/Heisenberg/data/trivial"
dataname = "Heisenberg/data"
D = 64
Lx = 20
Ly = 1
params = (J = 1, Δ = 1.0)

# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
@load "$(trivialname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI

# lsβ2 = 2lsβ[2:end]

# select_point = 8:24
# lsDWE = zeros(length(select_point))
# for (i,ind) in enumerate(select_point)
#     @load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(ind).jld2" data
#     # @show  sum(map(x -> sum(values(x)), values(data)))
#     lsDWE[i] = (lsβ2[ind]) ^ 2 *  sum(map(x -> sum(values(x)), values(data)))
# end

# lsDWE = (lsβ2) .^ 2 .*  map(x -> sum(sum.(values.(values(x)))), lsdata)

figsize = (width = 400 ,height = 250)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xlabel = L"T",ylabel = L"D_W^{E}",
title = "$(Ly)x$(Lx), D = $(D)")

scatterlines!(ax,1 ./ lsβ2, lsI .* lsβ2 .^ 2)
cT = range(0,0.15,100)
lines!(ax, cT, v(params.Δ) * cT, linestyle = :dash,color = :black)
xlims!(ax,0,0.5)
# ylims!(0,0.3)
resize_to_layout!(fig)
display(fig)



# filter(x -> 1 in x[1], )

# [lsdata[end][k] for k in keys(lsdata[end])]
# a = lsdata[end]
# [map(y -> a[] filter(x -> 1 ∉ x[1] && 1 ∉ x[2], collect(keys(a[k]))) for k in keys(a)])