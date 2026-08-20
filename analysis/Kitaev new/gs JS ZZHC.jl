using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/ZZHC"

Lx = 6
Ly = 3
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 256
lsHc = vcat(0.01:0.01:0.2)
lsJS = zeros(size(Latt),size(Latt),length(lsHc))
K = 1.0
proj = [0,1]
selectedpoints = 3:div(size(Latt),2)

lsJy = zeros(length(lsHc),2Lx)
lsSp = zeros(length(lsHc),size(Latt))

for (iHc,Hc) in enumerate(lsHc)
params = (K = K, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)
ĥ = normalize([Hx,Hy,Hz])
direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]
# edgepoints = vcat(1:Ly + 2, size(Latt)-Ly - 1:size(Latt))
edgepoints = []


Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata


for k in 1:3
    cinds = currentindex2(Js[k],h)
    for (jeff,(α,β)) in cinds
        for (j,l) in bonds[k]
            ij = ceil(Int64,j/2/Ly)
            il = ceil(Int64,l/2/Ly)
            ij ≠ il && continue
            lsJy[iHc,ij] += real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff * dot(relaVec(Latt,j,l),proj)
        end
    end
end
for i in 1:size(Latt)
    lsSp[iHc,i] = ĥ' * map(x -> real(obs[(x,)][(i,)]),["Sx","Sy","Sz"])
end

end

lsSc = mean(lsSp,dims = 2)[:]
figsize= (width=300,height=150)
fig = Figure()

ax = Axis(fig[1,1];
# xscale = log10,yscale=log10,
# xticks = [1,5,10,20,50] |> x -> (x,string.(x)),
title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D), K = $(K)", 
xlabel = L"H_{111}",ylabel= L"J_S",
figsize...)

ax2 = Axis(fig[2,1];
# xscale = log10,yscale=log10,
# xticks = [1,5,10,20,50] |> x -> (x,string.(x)),
# title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D), K = $(K)", 
xlabel = L"H_{111}",ylabel= L"S_{edge}",
figsize...)

# lsJyhalf = sum(lsJy[:,1:Lx] .- lsJy[:,Lx+1:end],dims = 2)[:]
lsJyhalf = abs.((lsJy[:,1] .- lsJy[:,end]) / Ly)
lsSedge = (mean(lsSp[:,1:Ly],dims = 2) + mean(lsSp[:,end-Ly+1:end],dims = 2))[:] /2

scatterlines!(ax,lsHc,(lsJyhalf))

scatterlines!(ax2,lsHc,lsSedge)

# model(x,p) = @. p[1]* + p[2]*x  + p[3]/x^3

# f = curve_fit(model,lsHc,lsJyhalf ,[0.1,0.1,-0.1])

# cx= range(extrema(lsHc)...,100)
# lines!(ax,cx,model(cx,f.param) ;color= :red)

ylims!(ax,0,0.02)
ylims!(ax2,0,0.5)

xlims!(ax,0,maximum(lsHc))
xlims!(ax2,0,maximum(lsHc))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_JS_Hc_$(Lx)x$(Ly)_$(D)_$(K).pdf",fig)
save("Kitaev new/figures/Nernst_JS_Hc_$(Lx)x$(Ly)_$(D)_$(K).png",fig)


# end
