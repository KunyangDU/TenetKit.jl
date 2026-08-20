using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/OXCHC"

Lx = 8
Ly = 2
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 256
lsHc = 0.02:0.02:0.2
lsJS = zeros(size(Latt),size(Latt),length(lsHc))
K = 1.0
proj = [0,1]
selectedpoints = 3:div(size(Latt),2)

lsJedge = zeros(length(lsHc))
lsSedge = zeros(length(lsHc))

cornerpoints = vcat(1:6Ly,size(Latt)-6Ly+1:size(Latt))
xedgepoints = filter(x -> length(neighbor(Latt,x))==2 && x ∈ cornerpoints,1:size(Latt))
yedgepoints = filter(x -> length(neighbor(Latt,x))==2 && x ∉ cornerpoints,1:size(Latt))
direction = [[1,0],[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2]]

for (iHc,Hc) in enumerate(lsHc)
params = (K = K, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)
ĥ = normalize([Hx,Hy,Hz])
# edgepoints = vcat(1:Ly + 2, size(Latt)-Ly - 1:size(Latt))



Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
@show Hc,lsEg[end]
obs = gsdata

for k in 1:3
    cinds = currentindex2(Js[k],h)
    for (j,l) in bonds[k]
        j ∉ yedgepoints && l ∉ yedgepoints && continue
        ans = 0.0
        for (jeff,(α,β)) in cinds
            ans += (real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff)
        end
        lsJedge[iHc] += abs(ans) / (2length(yedgepoints))
    end
end
for i in 1:size(Latt)
    if i in yedgepoints
        lsSedge[iHc] += dot(h, real(map(x -> obs[(x,)][(i,)],["Sx","Sy","Sz"]))) / length(yedgepoints)
    end
end


end
lsJedge

figsize= (width=300,height=150)
fig = Figure()

ax = Axis(fig[1,1];
# xscale = log10,yscale=log10,
# xticks = [1,5,10,20,50] |> x -> (x,string.(x)),
title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D), K = $(K)", 
xlabel = L"H_{111}",ylabel= L"J_S / H_{111}",
figsize...)


ax2 = Axis(fig[2,1];
# xscale = log10,yscale=log10,
xticks = 0:0.05:0.2 |> x -> (x,string.(x)),
# title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D), K = $(K)", 
xlabel = L"H_{111}",ylabel= L"S_{edge}",
figsize...)


scatterlines!(ax,lsHc,(lsJedge) ./ lsHc)
scatterlines!(ax2,lsHc,(lsSedge))

ylims!(ax,0,0.015)
ylims!(ax2,0,0.5)

xlims!(ax,0,maximum(lsHc))
xlims!(ax2,0,maximum(lsHc))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_JS_OXCHC_Hc_$(Lx)x$(Ly)_$(D)_$(K).pdf",fig)
save("Kitaev new/figures/Nernst_JS_OXCHC_Hc_$(Lx)x$(Ly)_$(D)_$(K).png",fig)
