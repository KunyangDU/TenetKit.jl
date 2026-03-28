using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,Interpolations
include("../../analysis/analysis.jl")
include("model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/ZZHC"
figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC"
tailname = ""

D = 2^9
Lx = 6
Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)
# @save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @show size(Latt)
J = 0.
Γ1′ = 0.
params23 = (J2 = 0., J3xy = 0., J3z = 0.0)



θ = 0.0
ϕ = pi / 2



lsHf = 0:0.04:0.8


K = 1
Γ  = 0.0
params1_Kitaev = (K = K,Γ = Γ)

lsSx = zeros(length(lsHf))
lsSy = zeros(length(lsHf))
lsSz = zeros(length(lsHf))

selected_point = Ly:size(Latt)-Ly

# lsM = zeros(length(lsHf))
projv = zeros(3)

for (i,Hf) in enumerate(lsHf)


Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,-1,0]/sqrt(2) + Hy * [1,1,-2]/sqrt(6) + Hz * [1,1,1]/sqrt(3);digits = 3)
# Hcx,Hcy,Hcz = Hx * [1,-1,0] + Hy * [1,1,-2] + Hz * [1,1,1]
projv[:] = [Hcx,Hcy,Hcz] |> x -> x/norm(x) 
params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

# println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
# @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
# lsE[i] = lsEg[end]
lsSx[i] = sum([gsdata["obs"][(("Sx",),)][((i,),)] for i in selected_point]) / length(selected_point)
lsSy[i] = sum([gsdata["obs"][(("Sy",),)][((i,),)] for i in selected_point]) / length(selected_point)
lsSz[i] = sum([gsdata["obs"][(("Sz",),)][((i,),)] for i in selected_point]) / length(selected_point)
end


# lsχ = [-(lsE[i] + lsE[i+2] - 2lsE[i+1])/(lsHf[i+2] - lsHf[i+1])/(lsHf[i+1] - lsHf[i]) for i in 1:length(lsE)-2]

# lsM = lsSx/sqrt(3) + lsSy/sqrt(3) + lsSz/sqrt(3)
lsM = projv[1] * lsSx + projv[2] * lsSy + projv[3] * lsSz


fig = Figure()
ax = Axis(fig[1,1])
# scatterlines!(ax,lsHf[1:end-2],lsχ)
scatterlines!(ax,lsHf,lsM)
interp_cubic = cubic_spline_interpolation(lsHf, lsM)
Hf_smooth = range((collect(extrema(lsHf)) - [0,0.01])...,200)
M_smooth = interp_cubic(Hf_smooth)
lines!(ax,Hf_smooth[1:end-1],[(M_smooth[i + 1] - M_smooth[i])/(Hf_smooth[i+1] - Hf_smooth[i]) for i in 1:length(Hf_smooth)-1];color = :red)

scatterlines!(ax,lsHf[1:end-1],[(lsM[i+1] - lsM[i])/(lsHf[i+1] - lsHf[i]) for i in 1:length(lsM)-1])

# ylims!(ax,0,1/2)
xlims!(ax,extrema(lsHf))

display(fig)
# lsχ



