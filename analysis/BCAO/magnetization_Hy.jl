using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/BCAO/data/yeesuan"

D = 2^7
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

select_point = div(size(Latt),Ly) |> x -> x+1:size(Latt)-x

lsHy = 0.0:0.1:1.0
lsSy = zeros(length(lsHy))
for (i,Hy) in enumerate(lsHy)
    params = (Hy = Hy, J1xy = -1, J1z = -0.36, Jpm = 0.023, Jzpm = -0.57, J2 = -0.032, J3xy = 0.26, J3z = 0.0078)
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
    lsSy[i] = [gsdata["Sy"][(i,)] for i in select_point] |> x -> sum(x) / length(x)
end

fig = Figure()

ax = Axis(fig[1,1])

scatterlines!(ax,lsHy,lsSy)
display(fig)
