using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../../analysis/analysis.jl")
include("model.jl")
Ly = 1
Lx = 61
dataname = "../codes/examples/J1J2chain/plateau/data"
@load "$dataname/Latt_$(Lx)x$(Ly).jld2" Latt
D = 128

lsk = range(-pi,pi,101)
lstk = [(k,0) for k in lsk]
lsH = 0:0.04:0.4
lsFSS = zeros(length(lsk),length(lsH))
for (i,H) in enumerate(lsH)
params = (J1 = -1, J2 = 0.5, J1xy = 0.2, Hx = 0.0, Hy = 0.0, Hz = H)
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata



FSxSx,FSySy,FSzSz = map(x -> FT2(getCorrMat(Latt,gsdata[(x,)],1/4;selected_point = 1:size(Latt),
# independent_data = gsdata[(tuple(x[1]),)]
),Latt,lstk),[("Sx","Sx"),("Sy","Sy"),("Sz","Sz")])
lsFSS[:,i] = FSxSx .+ FSySy .+ FSzSz
end

figsize = (height = 200,width = 300)

fig = Figure()
ax = Axis(fig[1,1];figsize...)

# lines!(ax,lsk / pi,FSS)

heatmap!(ax,lsk / pi,lsH,lsFSS,colorrange = (0,3))
lines!(ax,ones(2)/2,collect(extrema(lsH)),linestyle = :dash,color = :black)
lines!(ax,-ones(2)/2,collect(extrema(lsH)),linestyle = :dash,color = :black)
lines!(ax,ones(2)/3,collect(extrema(lsH)),linestyle = :dash,color = :red)
lines!(ax,-ones(2)/3,collect(extrema(lsH)),linestyle = :dash,color = :red)
resize_to_layout!(fig)
display(fig)