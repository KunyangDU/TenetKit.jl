using JLD2, CairoMakie, FiniteLattices, BenchmarkTools

Ly = 1
tailname = "SU2"
totalname = "examples/ShastrySuther/data"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
lsλ = vcat(0:0.05:0.6,0.61:0.01:0.8,0.85:0.05:1)
Lx = 4
Ly = 4

D = 2 ^ 9
J1 = 1
lsE = zeros(length(lsλ))

for i in eachindex(lsλ)
    λ = lsλ[i]
    params = (J1 = λ, J2 = 1)
    # @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    # lsE[i] = data["E"] / size(Latt)
    # @assert abs(data["σE"] / data["E"]) < 1e-4
    @load "../codes/$(totalname)/lsE_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" lsEg
    mE = sum(lsEg[end-2:end]) / 3
    σE = std(lsEg[end-2:end])
    @assert abs(σE / mE) < 1e-1 σE,mE
    lsE[i] = lsEg[end] / size(Latt)
end
figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,xlabel = L"J_1/J_2",ylabel = L"E/N",
title = "Ground state energy",
xticks = 0:0.1:1)

lines!(ax,0.66 * ones(2),collect(extrema(lsE));color = :grey)
lines!(ax,0.71 * ones(2),collect(extrema(lsE));color = :grey)
poly_points = ((extrema(lsE)) |> x -> Point2f[ (0.66, x[1]),  (0.71, x[1]), (0.71, x[2]), (0.66, x[2])])
poly!(ax, poly_points; color=(:grey,0.3))
scatterlines!(ax,lsλ,lsE,color = :blue,label = L"E/N")
insetsize = (width = 150,height = 100)

inset_ax = Axis(fig[1,1];insetsize...,
halign=0.25,    # 水平居中
valign=0.3,    # 垂直底部
backgroundcolor = :white,
xgridvisible=false,    # 关闭网格
ygridvisible=false,
xticks = ([0.66,0.71],[L"λ_{c1}",L"λ_{c2}"])
)

xlims!(inset_ax,0.6,0.8)
ylims!(inset_ax,-0.5,-0.33)
lines!(inset_ax,0.66 * ones(2),collect(extrema(lsE));color = :grey)
lines!(inset_ax,0.71 * ones(2),collect(extrema(lsE));color = :grey)
poly!(inset_ax, poly_points; color=(:grey,0.3))

scatterlines!(inset_ax,lsλ,lsE,color = :blue,label = L"E/N")


resize_to_layout!(fig)
display(fig)

save("ShastrySuther/figures/Eg_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).pdf",fig)
save("ShastrySuther/figures/Eg_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).png",fig)



