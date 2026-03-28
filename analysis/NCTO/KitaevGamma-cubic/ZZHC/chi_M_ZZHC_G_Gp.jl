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

# Hf = 0.0
selected_point = Ly:size(Latt)-Ly

lsM = zeros(length(lsHf))
for (i,Hf) in enumerate(lsHf)


Hx = Hf*sin(θ)*cos(ϕ)
Hy = Hf*sin(θ)*sin(ϕ)
Hz = Hf*cos(θ)

Hcx,Hcy,Hcz = Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3)

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)
params_H_name = (Hx = round.(Hcx;digits = 3), Hy = round.(Hcy;digits = 3), Hz = round.(Hcz;digits = 3))

params_Kitaev = merge(params1_Kitaev,params_H)
params_Kitaev_name = merge(params1_Kitaev,params_H_name)

# println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev_name).jld2" gsdata
# @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
S = map(x -> [gsdata["obs"][((x,),)][((i,),)] for i in selected_point] |> y -> sum(y)/length(y), ["Sx","Sy","Sz"])
lsM[i] =  dot( S,[Hcx,Hcy,Hcz]) / (Hf == 0 ? 1 : Hf)
end


lsχ = [(lsM[i+1] - lsM[i]) for i in 1:length(lsM)-1]



figsize = (width = 300,height = 150)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xlabel = L"h",ylabel = L"\chi_M",
title = "$(Ly)x$(Lx), D=$(D), χ_M")
scatterlines!(ax,lsHf[1:end-1],lsχ)
# scatterlines!(ax,lsHf,lsM)
xlims!(ax,extrema(lsHf))
resize_to_layout!(fig)
display(fig)
# lsχ

save("$(figurename)/χ_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).png",fig)
save("$(figurename)/χ_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).pdf",fig)
