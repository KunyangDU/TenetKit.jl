using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1"

Lx = 8
Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 256
lsHc = 0.1:0.1:1.0
lsJS = zeros(size(Latt),size(Latt),length(lsHc))
K = 1.0
proj = [0,1]
selectedpoints = 3:div(size(Latt),2)
Hc = 0.1

params = (K = K, Ha = 0.0, Hb = 0.0, Hc = Hc)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

direction = [[1,0],[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]
# edgepoints = vcat(1:Ly + 2, size(Latt)-Ly - 1:size(Latt))
edgepoints = []

Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata

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

@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
obs = data["obs"]
for k in 1:3
    cinds = currentindex2(Js[k],h)
    for (jeff,(α,β)) in cinds
        for (j,l) in bonds[k]
            M = real(obs[(Snames[α],Snames[β])][(j,l)])  * dot(proj,relaVec(Latt,j,l))
            lsIp[ycoordinatedict[j]] += M / 2
            lsIp[ycoordinatedict[l]] += M / 2
        end
    end
end


lsIph = map(x -> (mean(lsIp[1:div(length(ycoordinates),2),x])),eachindex(validinds))
lsIph = abs.(lsIph)
figsize = (width = 300, height = 200)

fig = Figure()
ax = Axis(fig[1,1];
# yscale = log10,
# xscale = log10,
xminorticksvisible = true, 
# xminorticks = IntervalsBetween(10),
yminorticksvisible = true,
xticks = 1:100,
yminorticks = IntervalsBetween(10),
xgridvisible = false, ygridvisible = false,
xminorgridvisible = false, yminorgridvisible = false,
title = "$(Ly)x$(Lx) Kitaev model, D = $(D)\n$(params)", 
xlabel = L"site",
ylabel = L"|J_E| / (L_x/2)",
figsize...)


lsid = [validinds[end]-1,]
for i in lsid
scatterlines!(ax, 1:length(ycoordinates), lsIp[:,i-1])
end
# scatterlines!(ax,1 ./ lsβeff,abs.(lsIph))

# xlims!(ax,1 / 200,10^(-0.))
# ylims!(ax,10^(-5.),10^(-1.))

resize_to_layout!(fig)
display(fig)
save("Kitaev new/figures/Nernst_single_site_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Kitaev new/figures/Nernst_single_site_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
