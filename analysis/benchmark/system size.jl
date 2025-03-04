using TensorKit,CairoMakie,JLD2,BenchmarkTools,LaTeXStrings,LsqFit

Ly = 1
D = 2^5
lsLx = 20:10:100

@load "../codes/examples/TrivialSpinlessFermion/data/bdata_D=$(D)_lsLx=$(lsLx)_Ly=$(Ly).jld" bdata

times,timeerr,memory,memoerr = eachcol(hcat(map(x -> [mean(x).time * 1e-9, std(x).time * 1e-9, mean(x).memory * 1e-3 / 2^20, std(x).memory * 1e-3 / 2^20],bdata["trial"])...)')
L = bdata["Ly"] .* bdata["Lx"]

func(x,p) = @.  p[1] * x^p[2]
result_time = curve_fit(func,L[4:end],times[4:end],[0,0] * 1.)
result_memory = curve_fit(func,L[4:end],memory[4:end],[0,0] * 1.)

fig = Figure()

figsize = (width = 450,height = 200)

axt = Axis(fig[1,1];xscale = log10,yscale=log10,
xticks = (10:10:100,string.(10:10:100)),
yticks = (0:5,string.(0:5)),
xminorticksvisible = true, xminorgridvisible = true,
xminorticks = IntervalsBetween(5),
yminorticksvisible = true, yminorgridvisible = true,
yminorticks = IntervalsBetween(5),
figsize...,
ylabel = L"\mathrm{Time}\ t\ /\ \mathrm{s}",
title="Benchmark by free fermion (energy error < 1e-4)")

axm = Axis(fig[2,1];xscale = log10,yscale=log10,
xticks = (10:10:100,string.(10:10:100)),
yticks = (0:5:20,string.(0:5:20)),
xminorticksvisible = true, xminorgridvisible = true,
xminorticks = IntervalsBetween(5),
yminorticksvisible = true, yminorgridvisible = true,
yminorticks = IntervalsBetween(5),
figsize...,
xlabel = L"\mathrm{System\ size}\ L",
ylabel = L"\mathrm{Memory}\ M\ /\ \mathrm{GiB}")

lines!(axt,L,func(L,result_time.param);color = :red,linestyle=:dash)
scatter!(axt,L,times)
errorbars!(axt,L,times,timeerr)
text!(axt,24,3;text = "log(t) / log(L) ~ $(round(result_time.param[2];digits=3))",color=:red,fontsize = 16)

lines!(axm,L,func(L,result_memory.param);color = :red,linestyle=:dash)
scatter!(axm,L,memory)
text!(axm,24,8;text = "log(M) / log(L) ~ $(round(result_memory.param[2];digits=3))",color=:red,fontsize = 16)

hidexdecorations!(axt;grid = false,ticks=false,minorgrid = false, minorticks = false)

resize_to_layout!(fig)
display(fig)

save("benchmark/figures/system size.pdf",fig)
save("benchmark/figures/system size.png",fig)

bdata["energyerror"]

