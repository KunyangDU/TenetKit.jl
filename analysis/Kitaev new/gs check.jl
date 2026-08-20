using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Kitaev/data1"

Lx = 8
Ly = 4
# Latt = ZZHoneyComb(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 128

direction = [[1,0],[1/2,sqrt(3)/2],[1/2,-sqrt(3)/2]]

fluxsites,fluxdirections,_ = getPBCflux(Latt,XCTria(Lx-1,Ly),direction;edge_shift = [0,sqrt(3)/2],flux_shift = [1,sqrt(3)/3],d = sqrt(3)/3)
lsflux = zeros(length(fluxsites))
lsS = zeros(3,size(Latt))

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 10.0)
Hx,Hy,Hz = params.Ha * [-2,1,1] / sqrt(6) + params.Hb * [0,1,-1] / sqrt(2) + params.Hc * [1,1,1] / sqrt(3)

edgepoints = vcat(1:4Ly + 2, size(Latt)-4Ly - 1:size(Latt))


Js = map(1:3) do i J = zeros(3,3); J[i,i] = params.K; J end
h = normalize([Hx,Hy,Hz])
bonds = getxyzbonds(Latt,direction)
Snames = ("Sx","Sy","Sz")

@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
obs = gsdata

for ifl in eachindex(fluxsites)
    dir = collect(fluxdirections[ifl])
    lsflux[ifl] = real(gsdata[Snames[dir]][fluxsites[ifl]])
end

for iL in 1:size(Latt)
    lsS[:,iL] = map(x -> real(obs[(x,)][(iL,)]), collect(Snames))
end

