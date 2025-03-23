using JLD2, CairoMakie, FiniteLattices



Ly = 1
tailname = "SU2"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
lsλ = 0:0.1:1
lsLx = [20,]
Lx = 20
N = Lx*Ly
D = 2 ^ 9
lsE = zeros(length(lsλ))

for i in eachindex(lsλ)
    λ = lsλ[i]
    params = (J1 = 1, J2 = λ)
    @load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-6
    lsE[i] = data["E"] / N

end
figsize = (width = 500,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...)
scatter!(ax,0.5,-3/8,color = :white,strokecolor = :red,strokewidth = 2,markersize = 14)

scatterlines!(ax,lsλ,lsE)

resize_to_layout!(fig)
display(fig)





