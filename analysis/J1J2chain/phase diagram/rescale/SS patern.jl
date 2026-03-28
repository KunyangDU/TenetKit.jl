using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../model.jl")
Ly = 1
tailname = "SU2"
totalname = "examples/J1J2chain/data/rescale"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
# lsλ = 0:0.1:1
# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1,1.2:0.2,2)
# lsλ = 0:0.05:1
lsλp = vcat(0.5:0.5:5)
# lsλp = vcat(0.2:0.2:2,2.5:0.5:5)
lsλ = vcat(-reverse(lsλp),0,lsλp)
lsLx = [20,]
Lx = 20
@load "../codes/$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9
J1 = 1

nnpair = neighbor(Latt;level = 1)
nnnpair = neighbor(Latt;level = 2)

lsE = zeros(length(lsλ))

figsize = (width = 300,height = 800)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
autolimitaspect = true,
title = "spin correlation pattern",
xlabel = "site",
ylabel = L"J_1/J_2",
xticks = 1:2:size(Latt) |> x -> (x,string.(x)),
yticks = (0:1:length(lsλ)-1)*2sqrt(3) .+ 1 .+ sqrt(3)/2 |> x -> (x,string.(lsλ)))

for (i,λ) in enumerate(lsλ)
    params = (J1 = λ, J2 = 1)
    @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-1
    nnbonds = [data["SS"][(i,i+1)] for i in 1:N-1]
    nnnbonds = [data["SS"][(i,i+2)] for i in 1:N-2]
    # bonds /= maximum(abs.(bonds))
    plotchainbond!(ax,Latt,nnpair,nnbonds,[0,i-1] * 2sqrt(3);
    colorlimit = 0.5 .* (-1,1),
    colormap = :bwr)
    plotchainbond!(ax,Latt,nnnpair,nnnbonds,[0,i-1] * 2sqrt(3);
    colorlimit = 0.5 .* (-1,1),
    colormap = :bwr)
    i == 1 && Colorbar(fig[1,2],limits = 0.5 .* (-1,1),colormap = :bwr, label = L"\langle \mathbf{S}_i\cdot\mathbf{S}_j \rangle")
    # text!(ax,0,i + 0.5;text = "$(λ)",fontsize = 12)
end
resize_to_layout!(fig)
display(fig)


save("J1J2chain/figures/rescale/SS patern_$(Lx)x$(Ly)_D=$(D)_$(tailname).pdf",fig)
save("J1J2chain/figures/rescale/SS patern_$(Lx)x$(Ly)_D=$(D)_$(tailname).png",fig)




