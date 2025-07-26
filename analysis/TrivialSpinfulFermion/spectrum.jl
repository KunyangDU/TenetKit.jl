using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../analysis/analysis.jl")
include("model.jl")

trivialname = "../codes/examples/TrivialSpinfulFermion/data"
dataname = "TrivialSpinfulFermion/data"
figurename = "TrivialSpinfulFermion/figure"
D = 2^7
Lx = 4
Ly = 1
β = 20

lskr = getk(Lx,Ly)
vpath,rpath,rnode = vrange([[-pi,0],[-pi/2,0],[0,0],[pi/2,0],[pi,0]],200)
lstk = Tuple.(vpath)

figsize = (height = 250,width = 400)

fig = Figure()
ax = Axis(fig[1,1];
title = "$(Ly)x$(Lx), D = $(D), β = $(β)",
ylabel = L"\beta G(k,\beta/2)",
xticks = (rnode,[L"-\pi",L"-\pi/2",L"0",L"\pi/2",L"\pi"]),figsize...)

lsμ₊ = 1:-0.2:0.2
lsμ₋ = -1:0.2:-0.2

for (i,μ) in enumerate(lsμ₊)
params = (t = 1,μ = μ,U=0)

@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@load "$(trivialname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
@load "$(trivialname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

FF = [data["G_$(i)"][(("F₊",), ("F₊⁺",))][((i,),(j,))] for i in 1:size(Latt),j in 1:size(Latt)]
FF = (FF .+ FF')/2
FTFF = FT2(FF,Latt,lstk)
lines!(ax,rpath,β * FTFF;linewidth = 2,color = (:red,i/length(lsμ₊)), label = "μ = $(μ)")
# lines!(ax, rpath[indr]*ones(2), collect(extrema(FTFF)))
end

for (i,μ) in enumerate(lsμ₋)
params = (t = 1,μ = μ,U=0)

@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@load "$(trivialname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
@load "$(trivialname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

FF = [data["G_$(i)"][(("F₊",), ("F₊⁺",))][((i,),(j,))] for i in 1:size(Latt),j in 1:size(Latt)]
FF = (FF .+ FF')/2
FTFF = FT2(FF,Latt,lstk)
lines!(ax,rpath,β * FTFF;linewidth = 2,color = (:blue,i/length(lsμ₊)), label = "μ = $(μ)")
end

μ = 0
params = (t = 1,μ = 0,U=0)

@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(params).jld2" data
@load "$(trivialname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
@load "$(trivialname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

FF = [data["G_$(i)"][(("F₊",), ("F₊⁺",))][((i,),(j,))] for i in 1:size(Latt),j in 1:size(Latt)]
FF = (FF .+ FF')/2
FTFF = FT2(FF,Latt,lstk)
lines!(ax,rpath,β * FTFF;linewidth = 4,color = :black, label = "μ = $(μ)")


_,indr₊ = findmin(x -> norm(x .- [lskr[3],0]),vpath)
_,indl₊ = findmin(x -> norm(x .+ [lskr[3],0]),vpath)
_,indr₋ = findmin(x -> norm(x .- [lskr[2],0]),vpath)
_,indl₋ = findmin(x -> norm(x .+ [lskr[2],0]),vpath)

lines!(ax, rpath[indr₊]*ones(2), [0,1];color = :red, linestyle = :dash)
lines!(ax, rpath[indl₊]*ones(2), [0,1];color = :red, linestyle = :dash)
lines!(ax, rpath[indr₋]*ones(2), [0,1];color = :blue, linestyle = :dash)
lines!(ax, rpath[indl₋]*ones(2), [0,1];color = :blue, linestyle = :dash)

Legend(fig[1,2],ax)

ylims!(ax,0,1)

xlims!(ax,extrema(rnode))
resize_to_layout!(fig)
display(fig)

save("$(figurename)/spectrum_$(Lx)x$(Ly)_$(D).pdf",fig)
save("$(figurename)/spectrum_$(Lx)x$(Ly)_$(D).png",fig)
