using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit
using Polynomials
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/XTRG/OHHC"

L = 2
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(L).jld2" Latt

D = 400
DS = 2^4
lsν = [0.0,0.25,0.5,0.75]
N = 20

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.2)
Hc = params.Hc
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction=[[1/2,sqrt(3)/2],[-1/2,sqrt(3)/2],[1,0]]

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")
proj = [0,1]
edgepoints = filter(x -> length(neighbor(Latt,x))==2,1:size(Latt))

lsJedge = zeros(length(lsν),N)
lsβeff = zeros(length(lsν),N)
for (iν,ν) in enumerate(lsν)
        @load "$(dataname)/lsβ_$(L)_$(D)_$(params)_$(ν).jld2" lsβ
for i in 1:N
    @load "$(dataname)/data_$(L)_$(D)_$(params)_$(ν)_$(i).jld2" data
    obs = data["obs"]
    # sp(x) = real.([[obs[("Sx",)][(x,)],obs[("Sy",)][(x,)],obs[("Sz",)][(x,)]]])
    # @show sp(32)
    for k in 1:3
        cinds = currentindex2(Js[k],h)
        for (j,l) in bonds[k]
            j ∉ edgepoints && l ∉ edgepoints && continue
            ans = 0.0
            for (jeff,(α,β)) in cinds
                ans += (real(obs[(Snames[α],Snames[β])][(j,l)]) * jeff)
            end
            lsJedge[iν,i] += abs(ans)
        end
    end
    lsβeff[iν,i] = lsβ[i]
end
end

lsJedge = lsJedge[:]
lsβeff = lsβeff[:]
figsize = (width = 400, height = 600)

fig = Figure()
ax = Axis(fig[1,1];
# yscale = log10,
xscale = log10,
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
xticks = [0.001,0.005,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
# yticks = -500:50:500,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "$(L) Kitaev model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"J_S",
figsize...)


scatterlines!(ax,1 ./ lsβeff,lsJedge)
lsκ = [- (lsJedge[i+2] - lsJedge[i]) / (log(lsβeff[i+2]) - log(lsβeff[i])) * lsβeff[i+1]  for i in 1:length(lsJedge)-2]
lsβeff1 = lsβeff[2:end-1]
# scatterlines!(ax,1 ./ lsβeff[2:end],- diff(lsJedge) ./ diff(log.(lsβeff)) .* lsβeff[2:end] .^ 2)
# scatterlines!(ax,1 ./ lsβeff1, lsκ .* lsβeff1,label = "Hc = $(Hc)")

fitlength = 17


xlims!(ax,1/2048,1)
# xlims!(ax2,1 / 4096,10^(0.))

# xlims!(ax,10^(0.),10^(2.))
# ylims!(ax,10^(-5.),10^(-1.))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_JS_XTRG_$(L)_$(D)_$(params)_$(proj).pdf",fig)
save("Kitaev new/figures/Nernst_single_JS_XTRG_$(L)_$(D)_$(params)_$(proj).png",fig)

# lsJy[1,:,end]