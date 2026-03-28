using CairoMakie,JLD2


dataname = "../codes/examples/Heisenberg/data/trivial"

D = 2^6
Lx = 64
Ly = 1
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

τ = 0.2
Nhot = -20
Tmax = 10

Nmax = 66

lsHz = 0:0.1:3

# params = (J=1.0, Δ = 1.0, Hz = 1.0)
# lsβ = vcat(2. .^ (-15:1:-1), 1:10)
lsβ = vcat((1.0 + τ) .^ (Nhot:1:-1), 1:τ:Tmax)
lsβ2 = lsβ[2:end]*2


Imax = 0.3
Imin = -0.03

# Is = zeros(length(lsHz),length(lsβ2))
# for (i,Hz) in enumerate(lsHz)
# params = (J=1.0, Δ = 1.0, Hz = Hz)

# @load "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
# Is[i,:] = [sum(map(x -> sum(values(x)),values(I)))/size(Latt) for I in lsI]
# end

@load "$(dataname)/Is_$(Lx)x$(Ly)_$(D)_$(length(lsHz))_$(length(lsβ2)).jld2" Is



figsize = (width = 300, height = 250)

fig = Figure()
ax = Axis(fig[1,1];
figsize...,
title = "$(Ly)x$(Lx) Heisenberg, D=$(D)",
xlabel = L"h_{[111]}",
ylabel = L"T/J",
xgridvisible = false,
ygridvisible = false
# xscale = log10,
# yscale = log10
)


# Is = filter(x -> (x > Imax ? Imax : x,Is)
# Is = clamp.(Is,Imin,Imax)

cmap = cgrad([:blue, :white, :red], [0, (-5e-4-Imin)/(Imax-Imin), 1])

# hm = heatmap!(ax, lsHz, 1 ./ lsβ2, Is;
# colormap = cmap,
# colorrange = (Imin,Imax)
# )

contourf!(ax,lsHz, 1 ./ lsβ2, Is;
colormap = cmap,
levels = range(Imin,Imax,150),
# linewidth = 0,
extendhigh = :red,
extendlow = :blue,
)

# lines!(ax, lsHz[lsinds .- 1], 1 ./ lsβ2)
ylims!(ax,0,0.5)
xlims!(ax,extrema(lsHz))


Colorbar(fig[1,2],hm;
label = L"\tilde{I}_2")

resize_to_layout!(fig)
display(fig)
save("Heisenberg/trivial/figures/SSE_ctf_$(Lx)x$(Ly)_$(D)_$(length(lsHz))_$(length(lsβ2)).pdf",fig)
save("Heisenberg/trivial/figures/SSE_ctf_$(Lx)x$(Ly)_$(D)_$(length(lsHz))_$(length(lsβ2)).png",fig)


