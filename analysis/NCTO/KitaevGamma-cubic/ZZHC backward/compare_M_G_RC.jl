
using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/NCTO/KitaevGamma-cubic/data/PCHC"
eddataname = "../codes/examples/NCTO/KitaevGamma-cubic/ED/data"

figurename = "NCTO/KitaevGamma-cubic/figures/ZZHC"
tailname = ""

D = 64
Lx = 3
Ly = 2

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt


lsHf = 0:0.05:0.8

ϕ =  0.0

K = 0.0
Γ = -1.0
params1_Kitaev = (K = K, Γ = Γ)
lsM = zeros(2,length(lsHf))
lsMed = zeros(2,length(lsHf))

lsE = zeros(2,length(lsHf))
lsEed = zeros(2,length(lsHf))

for (iθ,θ) in enumerate([0.0, 0.5 * pi])




# Hf = 0.0
selected_point = 1:size(Latt)




for (i,Hf) in enumerate(lsHf)

Hx = round(Hf*sin(θ)*cos(ϕ);digits = 3)
Hy = round(Hf*sin(θ)*sin(ϕ);digits = 3)
Hz = round(Hf*cos(θ);digits = 3)

Hcx,Hcy,Hcz = round.(Hx * [1,1,-2]/sqrt(6) + Hy * [1,-1,0]/sqrt(2) + Hz * [1,1,1]/sqrt(3);digits = 3)
# Hcx,Hcy,Hcz = Hx * [1,-1,0] + Hy * [1,1,-2] + Hz * [1,1,1]

params_H = (Hx = Hcx, Hy = Hcy, Hz = Hcz)

params_Kitaev = merge(params1_Kitaev,params_H)

# println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsEg
S = map(x -> [gsdata["obs"][((x,),)][((i,),)] for i in selected_point] |> y -> sum(y)/length(y), ["Sx","Sy","Sz"])
lsM[iθ,i] =  dot( S,[Hcx,Hcy,Hcz]) / (Hf == 0 ? 1 : Hf)
lsE[iθ,i] = lsEg[end]
end

params = (J = 0.0,K = 0.0,Γ = Γ,Γ′ = 0.0)

@load "$(eddataname)/eddata_diff_$(Lx)x$(Ly)_$(params)_θ=$(θ/pi)_ϕ=$(ϕ/pi).jld2" eddata
lsMed[iθ,:] = eddata["m"]
lsEed[iθ,:] = eddata["E"]

end


# lsχ = [-(lsM[i] + lsM[i+2] - 2lsE[i+1]) for i in 1:length(lsM)-2]






figsize = (height = 200,width = 300)
fig = Figure()

axE = Axis(fig[1,1];
xlabel = L"H", ylabel = L"E_g",
title = "$(Ly)x$(Lx)x2, D = $(D), $(params1_Kitaev)",
figsize...,
xticks = 0:0.2:1)

axErr = Axis(fig[2,1];
xlabel = L"H", ylabel = L"\Delta E_g",
# title = "$(Ly)x$(Lx)x2, D = $(D), $(params1_Kitaev)",
height = figsize.height/2,width = figsize.width,
yscale = log10,
xticks = 0:0.2:1)

ax = Axis(fig[3,1];
xlabel = L"H", ylabel = L"\langle \mathbf{S} \cdot \hat{\mathbf{H}} \rangle",
# title = "$(Ly)x$(Lx)x2, D = $(D), $(params1_Kitaev)",
figsize...,
xticks = 0:0.2:1)

scatter!(axE,lsHf,lsEed[1,:];strokecolor = :red,color = :white,strokewidth = 2,markersize = 14,label = L"\mathrm{ED}")
scatter!(axE,lsHf,lsEed[2,:];strokecolor = :red,color = :white,strokewidth = 2,markersize = 14)
scatterlines!(axE,lsHf,lsE[1,:];label = L"\mathrm{DMRG},\theta = 0.0\pi")
scatterlines!(axE,lsHf,lsE[2,:];label = L"\mathrm{DMRG},\theta = 0.5\pi")

scatterlines!(axErr,lsHf,abs.(lsE[1,:] - lsEed[1,:]);label = L"\mathrm{DMRG},\theta = 0.0\pi")
scatterlines!(axErr,lsHf,abs.(lsE[2,:] - lsEed[2,:]);label = L"\mathrm{DMRG},\theta = 0.5\pi")


scatter!(ax,lsHf,lsMed[1,:];strokecolor = :red,color = :white,strokewidth = 2,markersize = 14,label = L"\mathrm{ED}")
scatter!(ax,lsHf,lsMed[2,:];strokecolor = :red,color = :white,strokewidth = 2,markersize = 14)
scatterlines!(ax,lsHf,lsM[1,:];label = L"\mathrm{DMRG},\theta = 0.0\pi")
scatterlines!(ax,lsHf,lsM[2,:];label = L"\mathrm{DMRG},\theta = 0.5\pi")


hidexdecorations!(axE;ticks = false,grid = false)
hidexdecorations!(axErr;ticks = false,grid = false)

# scatterlines!(ax,lsHf,lsM)
# Legend(fig[1,2],ax)
axislegend(axE,position = :lb)
xlims!(axE,extrema(lsHf))
xlims!(axErr,extrema(lsHf))
xlims!(ax,extrema(lsHf))

ylims!(ax,0,1/2)
resize_to_layout!(fig)
display(fig)
# lsχ

save("$(figurename)/compare_ED_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).png",fig)
save("$(figurename)/compare_ED_$(Lx)x$(Ly)_$(D)_$(params1_Kitaev).pdf",fig)




