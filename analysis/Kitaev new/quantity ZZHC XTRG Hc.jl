using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1/XTRG/ZZHC"

Lx = 4
Ly = 3
Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 256
DS = 2^4
N = 20
lsν = [0.0,0.5]

figsize = (width = 400, height = 150)
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
xticks = [0.001,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
ylabel = L"C/L",
title = "$(Ly)x$(Lx) Kitaev model, D = $(D), K = $(params.K)", 
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xminorgridvisible = false,figsize...)
axS = Axis(fig[2,1];
xscale = log10,
# yscale = log10,
ylabel = L"S/L/\ln 2",
xticks = [0.001,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xminorgridvisible = false,figsize...)
axM = Axis(fig[3,1];
xscale = log10,
# yscale = log10,
ylabel = L"M/L",
xticks = [0.001,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xlabel = L"T",
xminorgridvisible = false,figsize...)


axdMdT = Axis(fig[4,1];
xscale = log10,
# yscale = log10,
ylabel = L"dM/dT/L",
xticks = [0.001,0.01,0.05,0.1,0.5,1.0] |> x -> (x,string.(x)),
xminorticksvisible = true, 
xminorticks = IntervalsBetween(10),
xgridvisible = false,
xlabel = L"T",
xminorgridvisible = false,figsize...)

for Hc in [0.1,0.2,0.3]
params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)
h = normalize([Hx,Hy,Hz])

lsβeff = zeros(length(lsν),N)
lsE = zeros(length(lsν),N)
lsF = zeros(length(lsν),N)
lsSp = zeros(ComplexF64,length(lsν),N)


for (iν,ν) in enumerate(lsν)
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)_$(ν).jld2" lsβ

for i in 1:N
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(ν)_$(i).jld2" data
    lsE[iν,i] = data["E"] / size(Latt)
    lsF[iν,i] = data["F"] / size(Latt)
    lsSp[iν,i] = sum([h' * [data["obs"][("Sx",)][(j,)],data["obs"][("Sy",)][(j,)],data["obs"][("Sz",)][(j,)]] for j in 1:size(Latt)])
    lsβeff[i] = lsβ[i]
end
end
lsE = lsE[:]
lsβeff = lsβeff[:]

scatterlines!(axC, 1 ./ lsβeff, lsE)
# scatterlines!(axC, 1 ./ lsβeff[2:end], diff(lsE) ./ diff(log.(1 ./ lsβeff)) .* lsβeff[2:end])
# scatterlines!(axS, 1 ./ lsβeff, (lsE .- lsF ) .* lsβeff ./ log(2))
# scatterlines!(axM, 1 ./ lsβeff, lsSp / size(Latt))
# scatterlines!(axdMdT, 1 ./ lsβeff[2:end], abs.(lsβeff[2:end] .^2 .* diff(lsSp) ./ diff(lsβeff) / size(Latt)))

end
for ax in [axC,axS,axM,axdMdT]
    xlims!(ax,1 / 4096,10. ^ (0.0))
end

# ylims!(axC,10. ^ (-2.),10. ^ (1.))
ylims!(axS,0,1)

resize_to_layout!(fig)
display(fig)

save("Kitaev new/figures/quantity_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/quantity_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
