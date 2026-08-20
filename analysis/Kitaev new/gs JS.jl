using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1"

Lx = 6
Ly = 2
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 256
lsHc = 0.1:0.1:1.0
lsJS = zeros(size(Latt),size(Latt),length(lsHc))
K = 1.0
proj = [0,1]
selectedpoints = 3:div(size(Latt),2)
for (i,Hc) in enumerate(lsHc)
params = (K = K, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction = [[1,0],[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]
# edgepoints = vcat(1:Ly + 2, size(Latt)-Ly - 1:size(Latt))
edgepoints = []

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata

JS = zeros(size(Latt),size(Latt))

for k in 1:3
    cinds = currentindex2(Js[k],h)
    for (jeff,(α,β)) in cinds
        for (j,l) in bonds[k]
            j in edgepoints && continue 
            l in edgepoints && continue
            JS[j,l] += real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff
        end
    end
end
lsJS[:,:,i] = JS
end

lsJShalf = zeros(length(lsHc))

for i in eachindex(lsHc)
    for (j,l) in neighbor(Latt)
        j ∉ selectedpoints && continue
        l ∉ selectedpoints && continue
        lsJShalf[i] += dot(relaVec(Latt,j,l), proj) * lsJS[j,l,i]
    end
end


figsize = (width = 400,height = 200)

fig = Figure()

ax = Axis(fig[1,1]; 
yscale = log10,
# autolimitaspect = true,figsize...,
# yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
# xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1)),
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\nK = $(K)", 
figsize...,
xlabel = L"B_c",ylabel = L"J^S_{\mathrm{total}}"
# xgridvisible = false,
# ygridvisible =false,
)

# plotLatt!(ax,Latt;site = true, sitelabel = false,bond = false)

scatterlines!(ax,lsHc,abs.(lsJShalf))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_Hc_JS_$(Lx)x$(Ly)_$(D)_$(K).pdf",fig)
save("Kitaev new/figures/Nernst_Hc_JS_$(Lx)x$(Ly)_$(D)_$(K).png",fig)


# end