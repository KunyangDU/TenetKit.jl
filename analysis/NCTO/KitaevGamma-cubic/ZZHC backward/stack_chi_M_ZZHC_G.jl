using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/ZZHC"
figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC/Gamma"
tailname = ""

D = 256
Lx = 3
Ly = 4

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt



lsHf = 0:0.005:0.1
# lsHf = 0:0.02:0.8


θ = 0.5 * pi 
ϕ = 0.5 * pi

K = -1.0
# Γ = 0.2

# lsΓ = vcat(0:0.02:0.08,0.12:0.04:0.2)
# lsΓ = vcat(0.0,0.1,0.15:0.05:0.3)
lsΓ = 0:0.05:0.3
lsχ = zeros(length(lsΓ),length(lsHf)-1)
lsHeff = zeros(length(lsHf))

for (iΓ,Γ) in enumerate(lsΓ)
params1_Kitaev = (K = K, Γ = Γ)

# Hf = 0.0
selected_point = Ly:size(Latt)-Ly

lsM = zeros(length(lsHf))

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
lsM[i] =  dot( S,[Hcx,Hcy,Hcz]) / (Hf == 0 ? 1 : Hf)
lsHeff[i] = sqrt(Hcx^2 + Hcy^2 + Hcz^2)
end


lsχ[iΓ,:] = [(lsM[i+1] - lsM[i])/(lsHeff[i+1]-lsHeff[i]) for i in 1:length(lsM)-1]

end

lscHf = lsHf[1:end-1]

figsize = (width = 200,height = 300)


fig = Figure()
ax = Axis(fig[1,1];
title = "$(Ly)x$(Lx)x4, D = $(D)",
ylabel = L"\chi_M",
xlabel = L"h",
xticks = 0:0.2:1,figsize...)
for (i,Γ) in enumerate(lsΓ)
scatterlines!(ax,lscHf,lsχ[i,:] .+ i * 20,label = "Γ=$(Γ)")
# scatterlines!(ax,lsHf,lsM)
end

Legend(fig[1,2],ax)

xlims!(ax,extrema(lsHf))

resize_to_layout!(fig)
display(fig)
# # lsχ
save("$(figurename)/stack_chi_M_$(Lx)x$(Ly)_$(D)_$(θ/pi)_$(ϕ/pi)_$(K).png",fig)
save("$(figurename)/stack_chi_M_$(Lx)x$(Ly)_$(D)_$(θ/pi)_$(ϕ/pi)_$(K).pdf",fig)

# lsM