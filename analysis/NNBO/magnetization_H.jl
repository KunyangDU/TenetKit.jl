using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/NNBO/data/H"

D = 3^4
Lx = 4
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

select_point = 1:size(Latt)

lsH = 0:0.2:2.8
lsSz = zeros(length(lsH))
for (i,H) in enumerate(lsH)
    params1_Kitaev = (J1 = -1, K1 = 0.6, Γ1 = 0, Γ1′ = 0)
    params3DH = (J3 = 1, D = -3,  H = H)
    paramsh = (h = 0.0,)

    params1 = let 
        v = collect(params1_Kitaev)
        v1 = PC2Y*v
        (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
    end

    params = merge(params1,params3DH,paramsh)
    params_Kitaev = merge(params1_Kitaev,params3DH,paramsh)
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
    lsSz[i] = [gsdata["Sz"][(i,)] for i in select_point] |> x -> sum(x) / length(x)
end

fig = Figure()
figsize = (width = 300,height = 150)

ax = Axis(fig[1,1];figsize...,
xlabel = L"H/J_1",
ylabel = L"\langle \mathbf{S}\cdot \mathbf{H} \rangle",
xticks = 0:0.4:3,
yticks = (0:1/3:1,[L"0",L"\frac{1}{3}",L"\frac{2}{3}",L"1"]))

scatterlines!(ax,lsH,lsSz)

resize_to_layout!(fig)
display(fig)
