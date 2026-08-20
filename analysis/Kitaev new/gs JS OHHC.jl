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
lsHc = 0.01:0.01:0.2

lsS = zeros(length(lsHc),size(Latt))
lsJedge = zeros(length(lsHc))
lsSedge = zeros(length(lsHc))

for (iHc,Hc) in enumerate(lsHc)
params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

@load "$(dataname)/gsdata_$(L)_$(D)_$(params).jld2" gsdata
obs = gsdata

for k in 1:3
    cinds = currentindex2(Js[k],h)
    for (j,l) in bonds[k]
        j ∉ edgepoints && l ∉ edgepoints && continue
        ans = 0.0
        for (jeff,(α,β)) in cinds
            ans += (real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff)
        end
        lsJedge[iHc] += abs(ans)
    end
end


for i in 1:size(Latt)
    lsS[iHc,i] = dot(h, real(map(x -> obs[(x,)][(i,)],["Sx","Sy","Sz"])))
    if i in edgepoints
        lsSedge[iHc] +=lsS[iHc,i] / length(edgepoints)
    end
end

end

figsize= (width=300,height=150)

fig = Figure()

ax = Axis(fig[1,1];
# xscale = log10,yscale=log10,
xticks = 0:0.05:0.2 |> x -> (x,string.(x)),
title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D), K = $(K)", 
# xlabel = L"H_{111}",
ylabel= L"J_S / H_{111}",
figsize...)

ax2 = Axis(fig[2,1];
# xscale = log10,yscale=log10,
xticks = 0:0.05:0.2 |> x -> (x,string.(x)),
# title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D), K = $(K)", 
xlabel = L"H_{111}",ylabel= L"S_{edge}",
figsize...)

scatterlines!(ax,lsHc,(lsJedge) ./ lsHc)

scatterlines!(ax2,lsHc,(lsSedge))

# model(x,p) = @. p[1]  + p[2]*x^p[3]

# f = curve_fit(model,lsHc,lsJyhalf ,randn(4))

# cx= range(extrema(lsHc)...,100)
# lines!(ax,cx,model(cx,f.param);color= :red)

ylims!(ax,0,0.8)
ylims!(ax2,0,0.5)

xlims!(ax,0,maximum(lsHc))
xlims!(ax2,0,maximum(lsHc))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_JS_OHHC_Hc_$(Lx)x$(Ly)_$(D)_$(K).pdf",fig)
save("Kitaev new/figures/Nernst_JS_OHHC_Hc_$(Lx)x$(Ly)_$(D)_$(K).png",fig)



