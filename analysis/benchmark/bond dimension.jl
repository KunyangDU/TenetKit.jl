using JLD2, CairoMakie, BenchmarkTools,LaTeXStrings, LsqFit

Lx = 20
Ly = 1
L = Lx*Ly
lsDs = [[200,400],[600,800]]
times = []
timeerr = []
memory = []
memoerr = []
for lsD in lsDs
    @load "../codes/examples/TrivialSpinlessFermion/data/bdata_lsD=$(lsD)_Lx=$(Lx)_Ly=$(Ly).jld" bdata
    times1,timeerr1,memory1,memoerr1 = eachcol(hcat(map(x -> [mean(x).time * 1e-9, std(x).time * 1e-9, mean(x).memory * 1e-3 / 2^20, std(x).memory * 1e-3 / 2^20],bdata["trial"])...)')

    push!(times,times1...)
    push!(timeerr,timeerr1...)
    push!(memory,memory1...)
    push!(memoerr,memoerr1...)
end
lsD = vcat(lsDs...)
func(x,p) = @.  p[1] * x^p[2]
result_time = curve_fit(func,lsD,times,[0,0] * 1.)
result_memory = curve_fit(func,lsD,memory,[0,0] * 1.)

fig = Figure()

figsize = (width = 450,height = 200)

axt = Axis(fig[1,1];xscale = log10,yscale=log10,
xticks = (100:100:4000,string.(100:100:4000)),
#yticks = (0:5,string.(0:5)),
xminorticksvisible = true, xminorgridvisible = true,
xminorticks = IntervalsBetween(5),
yminorticksvisible = true, yminorgridvisible = true,
yminorticks = IntervalsBetween(5),
figsize...,
ylabel = L"\mathrm{Time}\ t\ /\ \mathrm{s}",
title="Benchmark by free fermion (energy error < 1e-4)")

axm = Axis(fig[2,1];xscale = log10,yscale=log10,
xticks = (100:100:4000,string.(100:100:4000)),
#yticks = (0:5:20,string.(0:5:20)),
xminorticksvisible = true, xminorgridvisible = true,
xminorticks = IntervalsBetween(5),
yminorticksvisible = true, yminorgridvisible = true,
yminorticks = IntervalsBetween(5),
figsize...,
xlabel = L"\mathrm{Bond\ dimension}\ D",
ylabel = L"\mathrm{Memory}\ M\ /\ \mathrm{GiB}")

lines!(axt,lsD,func(lsD,result_time.param);color = :red,linestyle=:dash)
scatter!(axt,lsD,times)
errorbars!(axt,lsD,times,timeerr)
#text!(axt,24,3;text = "log(t) / log(L) ~ $(round(result_time.param[2];digits=3))",color=:red,fontsize = 16)

lines!(axm,lsD,func(lsD,result_memory.param);color = :red,linestyle=:dash)
scatter!(axm,lsD,memory)
#text!(axm,24,8;text = "log(M) / log(L) ~ $(round(result_memory.param[2];digits=3))",color=:red,fontsize = 16)

hidexdecorations!(axt;grid = false,ticks=false,minorgrid = false, minorticks = false)

resize_to_layout!(fig)
display(fig)

result_time.param[2],result_memory.param[2]

