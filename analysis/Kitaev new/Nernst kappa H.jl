using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1"

Lx = 8
Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 400
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 50

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
ylabel = L"\kappa / T",
figsize...)

for Hc in [0.1,0.5,1.0,2.0]
params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction = [[1,0],[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]
shift = [0,sqrt(3)/2]
edgepoints = vcat(1:4Ly + 2, size(Latt)-4Ly - 1:size(Latt))
# edgepoints = []
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
# lsβ2 = vcat(lsβ2,42:2:200)
validinds = 2:120
lsβeff = zeros(length(validinds))

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt;shift = shift, direction = direction)
Snames = ("Sx","Sy","Sz")
proj = [0,1]
diffrp = [dot(proj,collect(coordinate(Latt,i))) for i in 1:size(Latt)] |> x -> x * ones(size(Latt))' - ones(size(Latt)) * x'
ycdata = Dict()

ycoordinates,ycoordinatedict = let ycoordinates = Float64[]
    for i in 1:size(Latt)
        ycoordinates = union(ycoordinates, [round(coordinate(Latt,i)[1];digits = 8),])
    end
    sort!(ycoordinates)
    yd = Dict(y => i for (i,y) in enumerate(ycoordinates))
    ans = Dict()
    for i in 1:size(Latt)
        ans[i] = yd[round(coordinate(Latt,i)[1];digits = 8)]
    end
    ycoordinates,ans
end

lsIp = zeros(length(ycoordinates),length(validinds))

for (ii,i) in enumerate(validinds)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    obs = data["obs"]
    # sp(x) = real.([[obs[("Sx",)][(x,)],obs[("Sy",)][(x,)],obs[("Sz",)][(x,)]]])
    # @show sp(32)
    for k in 1:3
        cinds = currentindex2(Js[k],h)
        for (jeff,(α,β)) in cinds
            for (j,l) in bonds[k]
                M = real(obs[(Snames[α],Snames[β])][(j,l)]) * dot(proj,relaVec(Latt,j,l))
                lsIp[ycoordinatedict[j],ii] += M / 2
                lsIp[ycoordinatedict[l],ii] += M / 2
            end
        end
    end
    lsβeff[ii] = lsβ2[ii]
end

lsIph = map(x -> (mean(lsIp[1:div(length(ycoordinates),2),x])),eachindex(validinds))
lsκ = -lsβeff[2:end] .^ 2 .* diff(lsIph) ./ diff(lsβeff)
lsβdiff1 = lsβeff[2:end]
# lsIph = abs.(lsIph)


scatterlines!(ax,1 ./ lsβdiff1, abs.(lsκ .* lsβdiff1),label = "h = $(Hc)")
end

xlims!(ax,1 / 200,10^(-0.))
# ylims!(ax,10^(-5.),10^(-0.))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_kappa_$(Lx)x$(Ly)_$(D)_$(params)_$(proj).pdf",fig)
save("Kitaev new/figures/Nernst_single_kappa_$(Lx)x$(Ly)_$(D)_$(params)_$(proj).png",fig)
