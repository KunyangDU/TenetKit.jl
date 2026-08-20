using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra,LsqFit,Integrals
using Polynomials
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/XTRG/ZZHC"

FermionJS(x,p) =[FermionJS1(t,p) for t in x]
FermionJS1(x,p) = p[1]*solve(IntegralProblem((y,T) -> y^2/sqrt(y^2 +p[2]^2)*tanh(sqrt(y^2 + p[2]^2)/T/p[3]), (0.0, pi), x), QuadGKJL()).u


Lx = 6
Ly = 3
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 400
DS = 2^4
lsν = [0.0,0.25,0.5,0.75]
N = 20

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.2)
Hc = params.Hc
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction=[[sqrt(3)/2,1/2],[sqrt(3)/2,-1/2],[0,1]]


Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")
proj = [0,1]

lsJy = zeros(length(lsν),2Lx,N)
lsβeff = zeros(length(lsν),N)
for (iν,ν) in enumerate(lsν)
        @load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_$(ν).jld2" lsβ
for i in 1:N
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(ν)_$(i).jld2" data
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
                lsJy[iν,ij,i] += real(obs[(Snames[α],Snames[β])][(j,l)]) * dot(relaVec(Latt,j,l),proj) * jeff
            end
        end
    end
    lsβeff[iν,i] = lsβ[i]
end
end

lsJyhalf = sum(lsJy[:,1:Lx,:],dims = 2)
lsJyhalf = lsJyhalf[:]
lsβeff = lsβeff[:]
# lsJyhalf .-= lsJyhalf[end] - 1e-6
figsize = (width = 400, height = 200)

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
title = "$(Ly)x$(Lx)x4 Kitaev model, D = $(D)\n$(params)", 
xlabel = L"T",
ylabel = L"\kappa_N/T",
figsize...)

# scatter!(ax,1 ./ lsβeff,(lsJyhalf);
# color = :white, markersize = 12, strokewidth = 2,strokecolor = map(x -> x > 0 ? :black : :blue, lsJyhalf))

fitlength = 18
tail = 0

# scatter!(ax,1 ./ lsβeff[end-tail-fitlength+1:end-tail],lsJyhalf[end-tail-fitlength+1:end-tail];
# color = :white, markersize = 12, strokewidth = 2,strokecolor = :red)
# scatterlines!(ax,1 ./ lsβeff,lsJyhalf)
lsκ = [- (lsJyhalf[i+2] - lsJyhalf[i]) / (log(lsβeff[i+2]) - log(lsβeff[i])) * lsβeff[i+1]  for i in 1:length(lsJyhalf)-2]
lsβeff1 = lsβeff[2:end-1]
# scatterlines!(ax,1 ./ lsβeff[2:end],- diff(lsJyhalf) ./ diff(log.(lsβeff)) .* lsβeff[2:end] .^ 2)
# scatterlines!(ax,1 ./ lsβeff1, lsκ .* lsβeff1,label = "Hc = $(Hc)")


# model(x,p) = (@. p[1] + (p[2]*x^(3/2)*exp(-p[3]/x)))

# f = curve_fit(model,1 ./ lsβeff[end-tail-fitlength+1:end-tail],lsJyhalf[end-tail-fitlength+1:end-tail],[-0.01,100,0.001])
# FermionJS
f = curve_fit(FermionJS,1 ./ lsβeff[end-tail-fitlength+1:end-tail],lsJyhalf[end-tail-fitlength+1:end-tail],[-0.01,2.0,1000.,0.3])
# # [-0.001,10,-100,0.01]
cx = 10 .^ range(log10.([1/4096,0.01])...,100)
lines!(ax,cx[1:end-1],diff(FermionJS(cx,f.param)) ./ diff(cx) ./ cx[1:end-1],color = :red,linewidth = 3)

vlines!(ax,1 ./ lsβeff[[length(lsβeff)-tail-fitlength+1,length(lsβeff)-tail]],linestyle= :dash)
# ax2 = Axis(fig[2,1];
# # yscale = log10,
# xscale = log10,
# xminorticksvisible = true, 
# xminorticks = IntervalsBetween(10),
# yminorticksvisible = true,
# xticks = [0.001,0.005,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
# yminorticks = IntervalsBetween(10),
# xgridvisible = false, ygridvisible = false,
# xminorgridvisible = false, yminorgridvisible = false,
# # title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
# xlabel = L"T",
# ylabel = L"\kappa_S/T",
# figsize...)

# lines!(ax2,cx[2:end],diff(model(cx,f.param)) ./ diff(cx) ./ cx[2:end],color = :red)

xlims!(ax,1/4096,10 ^ (-1.5))
# xlims!(ax2,1 / 4096,10^(0.))

# xlims!(ax,10^(0.),10^(2.))
# ylims!(ax,-0.03,-0.00)

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_JS_XTRG_$(Lx)x$(Ly)_$(D)_$(params)_$(proj).pdf",fig)
save("Kitaev new/figures/Nernst_single_JS_XTRG_$(Lx)x$(Ly)_$(D)_$(params)_$(proj).png",fig)

f.param
# lsJy[1,:,end]