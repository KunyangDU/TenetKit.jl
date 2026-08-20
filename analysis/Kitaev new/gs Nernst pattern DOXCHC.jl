using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/DOXCHC"

Lx = 12
Ly = 1
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 200
Hc = 0.1
params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)


direction = [[1,0],[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2]]
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


figsize = (width = 80*Lx,height = 160*Ly*sqrt(3))

fig = Figure()

ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
xticks = (6.5 .+ 1.0 .* (0:Lx-1),string.(1:Lx)),
yticks = (sqrt(3)/6 + sqrt(3) .+ sqrt(3) * (0:Ly-1),string.(1:Ly)),
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
    c = log10(abs(JS[j,l]))
    arrow0!(ax,x,y,u,v;arrowsize = 0.5,linewidth = 2,color = get(colorschemes[:Greens],c,(-4,-2)))
end

Colorbar(fig[1,2],colormap = colorschemes[:Greens],colorrange= (-4,-2),
label = L"\log_{10}\ J_S")

plotLatt!(ax,Latt;bond = false,site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],h' * lsS .- mean(h' * lsS),(-0.01,0.01)),sitesize = 8*ones(size(Latt)))

Colorbar(fig[1,3],colormap = colorschemes[:bwr],colorrange= (-0.01,0.01),
label = L"h \cdot (S_i - \bar{S})")

# plotLatt!(ax,XCTria(Lx,4))

hidedecorations!(ax)

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
