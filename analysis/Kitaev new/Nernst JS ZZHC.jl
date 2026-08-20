using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/ZZHC"

Lx = 6
Ly = 3
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 400
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 100

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.1)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
# lsβ2 = vcat(lsβ2,42:2:200)
validinds = 2:74
lsβeff = zeros(length(validinds))

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction = direction)
Snames = ("Sx","Sy","Sz")
proj = [0,1]

lsJy = zeros(2Lx,length(validinds))

for (ii,i) in enumerate(validinds)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    obs = data["obs"]
    # sp(x) = real.([[obs[("Sx",)][(x,)],obs[("Sy",)][(x,)],obs[("Sz",)][(x,)]]])
    # @show sp(32)
    for k in 1:3
        cinds = currentindex2(Js[k],h)
        for (jeff,(α,β)) in cinds
            for (j,l) in bonds[k]
                ij = ceil(Int64,j/2/Ly)
                il = ceil(Int64,l/2/Ly)
                ij ≠ il && continue
                lsJy[ij,ii] += real(obs[(Snames[α],Snames[β])][(j,l)]) * dot(relaVec(Latt,j,l),proj)
            end
        end
    end
    lsβeff[ii] = lsβ2[ii]
end

lsJyhalf = sum(lsJy[1:Lx,:],dims = 1)[:]

figsize = (width = 300, height = 200)

fig = Figure()
ax = Axis(fig[1,1];
# yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
xticks = [0.01,0.05,0.1,0.5,1.0],
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"|J_S| / (L_x/2)",
figsize...)

scatterlines!(ax,1 ./ lsβeff,(lsJyhalf))

xlims!(ax,1 / 200,10^(0.))
# xlims!(ax,10^(0.),10^(2.))
# ylims!(ax,10^(-5.),10^(-1.))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_JS_$(Lx)x$(Ly)_$(D)_$(params)_$(proj).pdf",fig)
save("Kitaev new/figures/Nernst_single_JS_$(Lx)x$(Ly)_$(D)_$(params)_$(proj).png",fig)
lsJyhalf