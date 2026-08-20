using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/OHHC"

L = 2
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(L).jld2" Latt
# direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]]
direction=[[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2],[1,0]]
edgepoints = filter(x -> length(neighbor(Latt,x))==2,1:size(Latt))

D = 256
for Hc in 0.1
params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

# edgepoints = vcat(1:Ly + 2, size(Latt)-Ly - 1:size(Latt))

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

@load "$(dataname)/gsdata_$(L)_$(D)_$(params).jld2" gsdata
obs = gsdata

JS = zeros(size(Latt),size(Latt))

for k in 1:3
    cinds = currentindex2(Js[k],h)
    for (jeff,(α,β)) in cinds
        for (j,l) in bonds[k]
            # j in edgepoints && continue 
            # l in edgepoints && continue
            JS[j,l] += real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff
        end
    end
end


figsize = (height = 150*L, width = 150*L)

fig = Figure()

ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
# yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
# xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1)),
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
# xgridvisible = false,
# ygridvisible =false,
)

# plotLatt!(ax,Latt;site = true, sitelabel = false,bond = false)

lsS = zeros(3,size(Latt))
for i in 1:size(Latt)
    lsS[:,i] = real(map(x -> obs[(x,)][(i,)],["Sx","Sy","Sz"]))
end

mjs = maximum((abs.(JS)))
for k in 1:3,(j,l) in bonds[k]
    if JS[j,l] > 0
        x,y = coordinate(Latt,j)
        u,v = relaVec(Latt,j,l)
    else 
        x,y = coordinate(Latt,l)
        u,v = relaVec(Latt,l,j)
    end
    # c = ( abs(u) ≈ 0 ? (v > 0 ? -1 : 1) : (u > 0 ? 1 : -1) )* abs(JS[j,l])
    c = (abs(JS[j,l]))
    arrow0!(ax,x,y,u,v;arrowsize = 0.5,linewidth = 2,color = get(colorschemes[:Greens],c,(0.0,mjs)))
end

Colorbar(fig[1,2],colormap = colorschemes[:Greens],colorrange= (0.0,mjs),
label = L"\ln\ J_S")

plotLatt!(ax,Latt;bond = false,site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],h' * lsS,(-1/2,1/2)),sitesize = 8*ones(size(Latt)))

Colorbar(fig[1,3],colormap = colorschemes[:bwr],colorrange= (-1/2,1/2),
label = L"h \cdot S")

hidedecorations!(ax)

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_pattern_$(L)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/Nernst_single_pattern_$(L)_$(D)_$(params).png",fig)

end