using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/ZZHC"

Lx = 6
Ly = 3
Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 400
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 100

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.1)

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
lsβ2 = vcat(lsβ2,42:2:200)

validinds = 2:60
# lsI = zeros(length(validinds))
lsβeff = zeros(length(validinds))
lsE = zeros(length(validinds))
lsF = zeros(length(validinds))
lsS = zeros(ComplexF64,3,size(Latt),length(validinds))
lsWl = zeros(length(validinds),2Lx)
for (ii,i) in enumerate(validinds)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    lsE[ii] = data["E"] / size(Latt)
    lsF[ii] = data["F"] / size(Latt)
    lsS[1,:,ii] = [data["obs"][("Sx",)][(j,)] for j in 1:size(Latt)]
    lsS[2,:,ii] = [data["obs"][("Sy",)][(j,)] for j in 1:size(Latt)]
    lsS[3,:,ii] = [data["obs"][("Sz",)][(j,)] for j in 1:size(Latt)]
    lsβeff[ii] = lsβ2[ii]
    # obs = data["obs"]
    # for j in 1:2Lx
    #     sites = Tuple(2Ly*(j-1) .+ (1:2Ly))
    #     opnames = Tuple(repeat(["Sz",],2Ly))
    #     lsWl[ii,j] += real(obs[opnames][sites])
    # end
end
lsS = real.(lsS)
P = hcat([-2,1,1] / sqrt(6),[0,1,-1] / sqrt(2),[1,1,1] / sqrt(3))
h = P * normalize!([params.Ha,params.Hb,params.Hc])
lsSp = (h' * sum(lsS,dims = 2)[:,1,:])[:]

figsize = (width = 300, height = 150)
fig = Figure()
# axE = Axis(fig[1,1];
# xscale = log10,
# xticks = [0.01,0.05,0.1,0.5,1.0],
# xminorticksvisible = true, 
# xminorticks = IntervalsBetween(10),
# xgridvisible = false,
# ylabel = L"E/L",
# title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
# xminorgridvisible = false,figsize...)
axC = Axis(fig[1,1];
xscale = log10,
# yscale = log10,
xticks = [0.01,0.05,0.1,0.5,1.0],
ylabel = L"C/L",
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xminorgridvisible = false,figsize...)
axS = Axis(fig[2,1];
xscale = log10,
# yscale = log10,
ylabel = L"S/L/\ln 2",
xticks = [0.01,0.05,0.1,0.5,1.0],
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xminorgridvisible = false,figsize...)
axM = Axis(fig[3,1];
xscale = log10,
# yscale = log10,
ylabel = L"M/L",
xticks = [0.01,0.05,0.1,0.5,1.0],
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xlabel = L"T",
xminorgridvisible = false,figsize...)


axdMdT = Axis(fig[4,1];
xscale = log10,
# yscale = log10,
ylabel = L"dM/dT/L",
xticks = [0.01,0.05,0.1,0.5,1.0],
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xlabel = L"T",
xminorgridvisible = false,figsize...)

axWl = Axis(fig[5,1];
xscale = log10,
# yscale = log10,
ylabel = L"W_l",
xticks = [0.01,0.05,0.1,0.5,1.0],
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xlabel = L"T",
xminorgridvisible = false,figsize...)

# scatterlines!(axE, 1 ./ lsβeff, lsE)
scatterlines!(axC, 1 ./ lsβeff[2:end], -diff(lsE) .* lsβeff[2:end] .^ 2)
scatterlines!(axS, 1 ./ lsβeff, (lsE .- lsF ) .* lsβeff ./ log(2))
scatterlines!(axM, 1 ./ lsβeff, lsSp / size(Latt))
scatterlines!(axdMdT, 1 ./ lsβeff[2:end], abs.(lsβeff[2:end] .^2 .* diff(lsSp) ./ diff(lsβeff) / size(Latt)))
scatterlines!(axWl, 1 ./ lsβeff, abs.(mean(lsWl[:,2:end-1],dims = 2)[:]) * 2^(2Ly))
for ax in [axC,axS,axM,axdMdT,axWl]
    xlims!(ax,1 / 200,10. ^ (0.0))
end

# ylims!(axC,10. ^ (-2.),10. ^ (1.))
ylims!(axS,0,1)
ylims!(axWl,0,1)

resize_to_layout!(fig)
display(fig)

save("Kitaev new/figures/quantity_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/quantity_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
