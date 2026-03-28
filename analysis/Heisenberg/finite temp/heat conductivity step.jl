using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("../model.jl")

v(Δ::Number,J::Number = 1, tol::Number = 1e-8) = abs(Δ-1) < tol ? pi*J/2 : pi*J*sqrt(1 - Δ^2)/acos(Δ)/2


trivialname = "../codes/examples/Heisenberg/data/trivial"
dataname = "Heisenberg/data"
params = (J = 1, Δ = 0.5)


figsize = (width = 400 ,height = 250)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xlabel = L"T",ylabel = L"D_W^{E}",
title = "XXZ chain, Δ = $(params.Δ), D = $(D)"
# title = "$(Ly)x$(Lx), D = $(D)"
)

select_point = 1:21

# D = 2^7
Ly = 1

for (D,Lx) in [(2^7,10),(2^7,20),(2^7,40),(2^7,100),(200,256)]
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

lsI = zeros(length(select_point))
for i in eachindex(lsβ2)[select_point]
    @load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i+1).jld2" data
    lsI[i] = data["I"]
end

scatterlines!(ax,1 ./ lsβ2[select_point], lsI ,label = "$(Ly)x$(Lx)")
end

axislegend(ax,position = :rb)

cT = range(0,0.15,100)

lines!(ax, cT, v(params.Δ) * cT, linestyle = :dash,color = :black)
xlims!(ax,0,0.4)
ylims!(0,0.3)
resize_to_layout!(fig)
display(fig)

save("Heisenberg/figures/heat conductivity_$(Ly)_$(D)_$(params).png",fig)
save("Heisenberg/figures/heat conductivity_$(Ly)_$(D)_$(params).pdf",fig)


