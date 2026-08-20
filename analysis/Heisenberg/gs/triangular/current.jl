using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../../analysis/analysis.jl")
include("../../model.jl")

dataname = "../codes/examples/Heisenberg/data/triangular"

D = 256
Lx = 6
Ly = 6
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (J = 1.0 , Hx = 0.0, Hy = 0.1, Hz = 0.0)
h = normalize([params.Hx,params.Hy,params.Hz])
edgepoints = [1,size(Latt)]

Snames = ("Sx","Sy","Sz")


@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata

JS = zeros(size(Latt),size(Latt))


cinds = currentindex2(diagm(ones(3) * params.J),h)
for (jeff,(α,β)) in cinds
    for (j,l) in neighbor(Latt)
        @show real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff
        JS[j,l] += real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff
    end
end

lsS = zeros(3,size(Latt))
for i in 1:size(Latt)
    lsS[:,i] = real(map(x -> obs[(x,)][(i,)],["Sx","Sy","Sz"]))
end

figsize = (width = 70*(Lx),height = 70*(Ly + 1)*sqrt(3)/2)

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
    arrow0!(ax,x,y,u,v;arrowsize = 0.3,linewidth = 3,color = get(colorschemes[:Greens],c,(0.0,mjs)))
end

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
    text!(ax,x + u/2, y + v/2,text = "$(round(c / mjs;digits = 3))",fontsize = 8)
end

plotLatt!(ax,Latt;bond = false,site = true,sitelabel = false,sitecolor = get(colorschemes[:bwr],h' * lsS,(-1/2,1/2)))
Colorbar(fig[1,2],colorrange = (0.0,mjs),colormap = colorschemes[:Greens],label = L"J_S")
hidedecorations!(ax)

resize_to_layout!(fig)
display(fig)
save("Heisenberg/gs/figures/triangular/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/gs/figures/triangular/Nernst_single_pattern_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


JSS = (JS - JS')

rd = [dot([0,1],relaVec(Latt,i,j)) for i in 1:size(Latt),j in 1:size(Latt)]
sum(JSS .* rd)