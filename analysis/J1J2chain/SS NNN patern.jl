using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("model.jl")
Ly = 1
tailname = "SU2"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
# lsλ = 0:0.1:1
# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1,1.2:0.2,2)
lsλ = 0:0.1:2
lsLx = [20,]
Lx = 20
@load "../codes/examples/J1J2chain/data/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 10
J1 = 1

nnpair = neighbor(Latt;level = 2)
lsE = zeros(length(lsλ))

figsize = (width = 500,height = 300)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "Next nearest neighbor correlation",
xlabel = "site",
ylabel = L"J2/J1",
xticks = 1:1:size(Latt) |> x -> (x,string.(x)),
yticks = 1:1:length(lsλ) |> x -> (x,string.(lsλ))
)

for (i,λ) in enumerate(lsλ)
    params = (J1 = J1, J2 = λ)
    @load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-1
    bonds = zeros(N-2)
    for i in 1:N-2
        bonds[i] = data["SS"][(i,i+2)]
    end
    # bonds /= maximum(abs.(bonds))
    plotbond!(ax,nnpair,bonds,[0,i-1];
    colorlimit = 0.5 .* (-1,1),
    colormap = :bwr)
    i == 1 && Colorbar(fig[1,2],limits = 0.5 .* (-1,1),colormap = :bwr, label = L"\langle \mathbf{S}_i\cdot\mathbf{S}_j \rangle")
    # text!(ax,0,i + 0.5;text = "$(λ)",fontsize = 12)
end
resize_to_layout!(fig)
display(fig)


save("J1J2chain/figures/NNN patern_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).pdf",fig)
save("J1J2chain/figures/NNN patern_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).png",fig)




