using JLD2,CairoMakie,LaTeXStrings,FiniteLattices
Lx = 11
Ly = 1
D = 30
Latt = YCSqua(Lx,Ly)

@load "../codes/examples/Ising/data/data_evolve_D=$(D)_$(Lx)x$(Ly).jld2" data

lst = data["lst"]
Szm = data["Szm"]

figsize = (width=300,height =200)

fig = Figure()
ax1 = Axis(fig[1,1];figsize...,ylabel="site",yticks = 1:size(Latt))


heatmap!(ax1,lst,1:size(Latt),Szm[:,:,1])


axislegend(ax3;position=:rb)


resize_to_layout!(fig)
display(fig)

save("Ising/figures/ising_evolve_D=$(D)_$(Lx)x$(Ly).pdf",fig)
save("Ising/figures/ising_evolve_D=$(D)_$(Lx)x$(Ly).png",fig)


