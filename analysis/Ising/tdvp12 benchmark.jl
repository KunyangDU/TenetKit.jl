using JLD2,CairoMakie,LaTeXStrings,FiniteLattices
Lx = 11
Ly = 1
D = 30
Latt = YCSqua(Lx,Ly)

@load "../codes/examples/Ising/data/data_D=$(D)_$(Lx)x$(Ly).jld2" data

lst = data["lst"]
Szm = data["Szm"]
error12 = data["error12"]
errorlr1 = data["errorlr1"]
errorlr2 = data["errorlr2"]

figsize = (width=300,height =200)

fig = Figure()
ax1 = Axis(fig[1,1];figsize...,ylabel="site",yticks = 1:size(Latt))
ax2 = Axis(fig[2,1];figsize...,xlabel=L"tJ",ylabel="site",yticks = 1:size(Latt))
ax3 = Axis(fig[1,2];figsize...,ylabel="asymmetric error")
ax4 = Axis(fig[2,2];figsize...,xlabel=L"tJ",ylabel="TDVP1 - TDVP2")

heatmap!(ax1,lst,1:size(Latt),Szm[:,:,1])
heatmap!(ax2,lst,1:size(Latt),Szm[:,:,2])
scatterlines!(ax3,lst,errorlr1,label="TDVP1")
scatterlines!(ax3,lst,errorlr2,label="TDVP2")
scatterlines!(ax4,lst,error12)

hidexdecorations!(ax1;ticks=false)
hidexdecorations!(ax3;ticks=false,grid=false)

# axislegend(ax3;position=:rb)

resize_to_layout!(fig)
display(fig)

save("Ising/figures/ising_tdvp12_benchmark_D=$(D)_$(Lx)x$(Ly).pdf",fig)
save("Ising/figures/ising_tdvp12_benchmark_D=$(D)_$(Lx)x$(Ly).png",fig)



