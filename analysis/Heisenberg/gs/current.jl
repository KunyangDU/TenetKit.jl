using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/data/trivial"


D = 128
Lx = 16
Ly = 1
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (Jxy=0.0, Jz = 1.0, Hx = 0.0, Hy = 0.1, Hz = 0.0)
h = normalize([params.Hx,params.Hy,params.Hz])
edgepoints = [1,size(Latt)]

Snames = ("Sx","Sy","Sz")


@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata

JS = zeros(size(Latt),size(Latt))


cinds = currentindex2(diagm([params.Jxy, params.Jxy, params.Jz]),h)
for (jeff,(α,β)) in cinds
    for (j,l) in neighbor(Latt)
        JS[j,l] += real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff
    end
end

lsS = zeros(3,size(Latt))
for i in 1:size(Latt)
    lsS[:,i] = real(map(x -> obs[(x,)][(i,)],["Sx","Sy","Sz"]))
end

figsize = (width = 60*(Lx),height = 60*(Ly + 1)*sqrt(3)/2)

fig = Figure()

ax = Axis(fig[1,1]; autolimitaspect = true,figsize...,
yticks = (sqrt(3)*7/12 .+ sqrt(3)/2 .* (0:2Ly-1),string.(1:2Ly)),
xticks = (1/2 .+ (1:1/2:Lx+1),string.(1:2Lx+1)),
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
# xgridvisible = false,
# ygridvisible =false,
)

# plotLatt!(ax,Latt;site = true, sitelabel = false,bond = false)

mjs = maximum((abs.(JS)))
for (j,l) in neighbor(Latt)
    # j in edgepoints && continue 
    # l in edgepoints && continue
    if JS[j,l] > 0
        x,y = coordinate(Latt,j)
        u,v = relaVec(Latt,j,l)
    else 
        x,y = coordinate(Latt,l)
        u,v = relaVec(Latt,l,j)
    end
    c = abs(JS[j,l])
    arrow0!(ax,x,y,u,v;arrowsize = 0.5,linewidth = 3,color = get(colorschemes[:Greens],c,(0.0,mjs)))
end

plotLatt!(ax,Latt;bond = false,site = true,sitelabel = true,sitecolor = get(colorschemes[:bwr],lsS[3,:],(-1/2,1/2)))
Colorbar(fig[1,2],colorrange = (0.0,mjs),colormap = colorschemes[:Greens])
hidedecorations!(ax)

resize_to_layout!(fig)
display(fig)
# save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
# save("Kitaev new/figures/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


