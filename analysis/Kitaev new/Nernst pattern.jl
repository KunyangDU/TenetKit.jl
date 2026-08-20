using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1"

Lx = 8
Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 400
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 50

params = (K = -1.0, Ha = 0.0, Hb = 0.0, Hc = 0.1)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction = [[1,0],[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]
edgepoints = vcat(1:4Ly + 2, size(Latt)-4Ly - 1:size(Latt))
# edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2


Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

indβ = 30

@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(indβ).jld2" data
obs = data["obs"]

JS = zeros(size(Latt),size(Latt))

for k in 1:3
    cinds = currentindex2(Js[k],h)
    for (jeff,(α,β)) in cinds
        for (j,l) in bonds[k]
            JS[j,l] += real(obs[(Snames[α],Snames[β])][(j,l)])
        end
    end
end


figsize = (width = 60*(Lx),height = 60*(Ly + 1)*sqrt(3)/2)

fig = Figure()

ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1)),
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params), β = $(lsβ2[indβ])", 
# xgridvisible = false,
# ygridvisible =false,
)

# plotLatt!(ax,Latt;site = true, sitelabel = false,bond = false)

mjs = maximum(abs.(JS))
for k in 1:3,(j,l) in bonds[k]
    if JS[j,l] > 0
        x,y = coordinate(Latt,l)
        u,v = relaVec(Latt,l,j)
    else 
        x,y = coordinate(Latt,j)
        u,v = relaVec(Latt,j,l)
    end
    c = ( abs(u) ≈ 0 ? (v > 0 ? -1 : 1) : (u > 0 ? 1 : -1) )* abs(JS[j,l])
    arrow0!(ax,x,y,u,v;arrowsize = 0.5,linewidth = 2,color = get(colorschemes[:bwr],c,(-mjs,mjs)))
end

Colorbar(fig[1,2],colormap = colorschemes[:bwr],colorrange= (-mjs,mjs),
label = L"J_S")


hidedecorations!(ax)

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


