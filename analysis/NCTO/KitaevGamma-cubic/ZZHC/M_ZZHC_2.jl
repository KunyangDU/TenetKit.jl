using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/ZZHC"
figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC"
tailname = ""

D = 256
Lx = 3
Ly = 4

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt


lsHf = 0:0.08:0.8


lsM = zeros(2,length(lsHf))
for (iθ,θ) in enumerate([0.0, 0.5 * pi])
ϕ = pi / 2

K = 0.0
Γ = -1.0

params1_Kitaev = (K = K, Γ = Γ)

# Hf = 0.0
selected_point = 1:size(Latt)




for (i,Hf) in enumerate(lsHf)

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)
# Hcx,Hcy,Hcz = Hx * [1,-1,0] + Hy * [1,1,-2] + Hz * [1,1,1]

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

# println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
# @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
S = map(x -> [gsdata["obs"][((x,),)][((i,),)] for i in selected_point] |> y -> sum(y)/length(y), ["Sx","Sy","Sz"])
lsM[iθ,i] =  dot( S,[Hcx,Hcy,Hcz]) / (Hf == 0 ? 1 : Hf)
end
end


# lsχ = [-(lsM[i] + lsM[i+2] - 2lsE[i+1]) for i in 1:length(lsM)-2]




figsize = (height = 200,width = 300)
fig = Figure()
ax = Axis(fig[1,1];
xlabel = L"H", ylabel = L"\langle \mathbf{S} \cdot \hat{\mathbf{H}} \rangle",
title = "$(Ly)x$(Lx)x4, D = $(D), $(params1_Kitaev)",
figsize...,
xticks = 0:0.2:1)
scatterlines!(ax,lsHf,lsM[1,:];label = L"\theta = 0.0\pi")
scatterlines!(ax,lsHf,lsM[2,:];label = L"\theta = 0.5\pi")

# scatterlines!(ax,lsHf,lsM)
axislegend(ax,position = :cc)
ylims!(ax,0,1/2)
xlims!(ax,extrema(lsHf))
resize_to_layout!(fig)
display(fig)
# lsχ

lsM