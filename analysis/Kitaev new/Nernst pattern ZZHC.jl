using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/ZZHC"

Lx = 4
Ly = 3
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 128
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 50

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.2)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]
edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2


Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

indβ = 41

@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(indβ).jld2" data
obs = data["obs"]

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

lsS = zeros(3,size(Latt))
for i in 1:size(Latt)
    lsS[:,i] = real(map(x -> obs[(x,)][(i,)],["Sx","Sy","Sz"]))
end

figsize = (width = 30*Lx*3,height = 30*(Ly+1/2)*sqrt(3))

fig = Figure()

ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1)),
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params), β = $(lsβ2[indβ])", 
# xgridvisible = false,
# ygridvisible =false,
)

# plotLatt!(ax,Latt;site = true, sitelabel = false,bond = false)

mjs = maximum(log10.(abs.(JS)))
for k in 1:3,(j,l) in bonds[k]
    if JS[j,l] > 0
        x,y = coordinate(Latt,j)
        u,v = relaVec(Latt,j,l)
    else 
        x,y = coordinate(Latt,l)
        u,v = relaVec(Latt,l,j)
    end
    # c = ( abs(u) ≈ 0 ? (v > 0 ? -1 : 1) : (u > 0 ? 1 : -1) )* abs(JS[j,l])
    c = log10(abs(JS[j,l]))
    arrow0!(ax,x,y,u,v;arrowsize = 0.5,linewidth = 2,color = get(colorschemes[:Greens],c,(-5,1)))
end

Colorbar(fig[1,2],colormap = colorschemes[:Greens],colorrange= (-8,-1),
label = L"\ln\ J_S")

plotLatt!(ax,Latt;bond = false,site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],h' * lsS,(-1/2,1/2)),sitesize = 8*ones(size(Latt)))

Colorbar(fig[1,3],colormap = colorschemes[:bwr],colorrange= (-1/2,1/2),
label = L"h \cdot S")


hidedecorations!(ax)

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


