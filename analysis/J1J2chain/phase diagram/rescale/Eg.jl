using JLD2, CairoMakie, FiniteLattices

Ly = 1
tailname = "SU2"
totalname = "examples/J1J2chain/data/rescale"
# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
lsλp = vcat(0.2:0.2:2,2.5:0.5:5)
lsλ = vcat(-reverse(lsλp),0,lsλp)
lsLx = [20,]
Lx = 20
N = Lx*Ly
D = 2 ^ 9
lsE = zeros(length(lsλ))

for i in eachindex(lsλ)
    λ = lsλ[i]
    params = (J1 = λ, J2 = 1)
    @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-1
    lsE[i] = data["E"] / N
end
figsize = (width = 400,height = 300)

fig = Figure()
ax = Axis(fig[1,1];figsize...,xlabel = L"J_1/J_2",ylabel = L"E/NJ_i")

# lines!(ax,collect(extrema(lsλ)),[lsE[1],lsE[1]],color = :grey,linestyle = :dash,label = L"\mathrm{ED}")
scatter!(ax,2,-3/8,color = :white,strokecolor = :red,strokewidth = 2,markersize = 14,label = L"\mathrm{Dimer}")
scatter!(ax,2,-3/4,color = :white,strokecolor = :red,strokewidth = 2,markersize = 14)
scatterlines!(ax,lsλ,lsE ,color = :green,label = L"E/NJ_2")
scatterlines!(ax,lsλ[eachindex(lsλp)],lsE[eachindex(lsλp)] ./ abs.(lsλ[eachindex(lsλp)]),color = :red,label = L"\mathrm{FA}\ E/NJ_1")
scatterlines!(ax,lsλ[length(lsλp) + 1 .+ eachindex(lsλp)],lsE[length(lsλp) + 1 .+ eachindex(lsλp)] ./ abs.(lsλ[length(lsλp) + 1 .+ eachindex(lsλp)]),color = :blue,label = L"\mathrm{AA}\ E/NJ_1")

# scatterlines!(ax,lsλ[2:end],(lsE ./ lsλ)[2:end],color = :green,label = L"E/NJ_2")
ylims!(ax,-0.9,-0.1)
axislegend(ax,position = :lb)
resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/rescale/Eg_$(Lx)x$(Ly)_D=$(D)_$(tailname).pdf",fig)
save("J1J2chain/figures/rescale/Eg_$(Lx)x$(Ly)_D=$(D)_$(tailname).png",fig)




