using JLD2, CairoMakie, FiniteLattices

Ly = 1
tailname = "SU2"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
lsλ = 0:0.1:2
lsLx = [20,]
Lx = 20
N = Lx*Ly
D = 2 ^ 9
J1 = 1
lsE = zeros(length(lsλ))

for i in eachindex(lsλ)
    λ = lsλ[i]
    params = (J1 = J1, J2 = λ)
    @load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-1
    lsE[i] = data["E"] / N
end
figsize = (width = 400,height = 300)

fig = Figure()
ax = Axis(fig[1,1];figsize...,xlabel = L"J_2/J_1",ylabel = L"E/NJ_i")

lines!(ax,collect(extrema(lsλ)),[lsE[1],lsE[1]],color = :grey,linestyle = :dash,label = L"\mathrm{ED}")
scatter!(ax,0.5,-3/8,color = :white,strokecolor = :red,strokewidth = 2,markersize = 14,label = L"\mathrm{Dimer}")
scatter!(ax,0.5,-3/4,color = :white,strokecolor = :red,strokewidth = 2,markersize = 14)

scatterlines!(ax,lsλ,lsE,color = :blue,label = L"E/NJ_1")
scatterlines!(ax,lsλ[2:end],(lsE ./ lsλ)[2:end],color = :green,label = L"E/NJ_2")
ylims!(ax,-0.9,-0.3)
axislegend(ax,position = :lb)
resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/Eg_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).pdf",fig)
save("J1J2chain/figures/Eg_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).png",fig)




