using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/ZZHC"
figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC"
tailname = ""

D = 2^8
Lx = 8
Ly = 2
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@show size(Latt)



lsHf = 0:0.04:0.8


θ = pi / 2
ϕ = pi / 2

K = 1
Γ = 0.0

params1_Kitaev = (K = K, Γ = Γ)

# Hf = 0.0


lsE = zeros(length(lsHf))

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
lsSE[i] = gsdata["SE"] |> x -> sum(x)/length(x)
end




interp_cubic = cubic_spline_interpolation(lsHf, lsSE)
Hf_smooth = range((collect(extrema(lsHf)) - [0,0.01])...,200)
SE_smooth = interp_cubic(Hf_smooth)



fig = Figure()
ax = Axis(fig[1,1])
# lines!(ax,Hf_smooth[1:end-1],[-0.5*(SE_smooth[i + 1] - SE_smooth[i])/(Hf_smooth[i+1] - Hf_smooth[i]) for i in 1:length(Hf_smooth)-1];color = :red)

scatterlines!(ax,lsHf[1:end-1],-[lsSE[i+1] - lsSE[i] for i in 1:length(lsHf)-1])
scatterlines!(ax,lsHf,lsSE)

# ylims!(ax,0,1/2)
xlims!(ax,extrema(lsHf))

display(fig)
# lsχ



