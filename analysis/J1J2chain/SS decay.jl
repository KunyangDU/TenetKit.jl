using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("model.jl")
Ly = 1
tailname = "SU2"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
# lsλ = 0:0.1:1
# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1,1.2:0.2,2)
lsλ = 1:0.2:2
lsLx = [20,]
Lx = 20
@load "../codes/examples/J1J2chain/data/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

figsize = (width = 500,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...)

for (i,λ) in enumerate(lsλ)
    params = (J1 = 1, J2 = λ)
    @load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-6
    bonds = zeros(N-1)
    for j in 1:N-1
        bonds[j] = data["SS"][(1,j+1)]
    end
    bonds = abs.(bonds)
    scatterlines!(ax,1:N-1,bonds,color = (:red,i / (N-1)))
end
resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/SSdecay.pdf",fig)
save("J1J2chain/figures/SSdecay.png",fig)






