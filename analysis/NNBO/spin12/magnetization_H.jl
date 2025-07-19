include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/NNBO/spin12/data/H"
figurename = "NNBO/spin12/figures"

D = 2^6
Lx = 4
Ly = 4
params1_1_Kitaev = (J1 = -1, K1 = 0.6, Γ1 = 0, Γ1′ = 0)
params1_3_JDH = (J3 = 1, D = 3)

lsH = 0:0.2:2.8
lsSz = zeros(length(lsH))

for (i,H) in enumerate(lsH)
    paramsh = (h=0.0, H = H)

    params1_cry = _Cub2Cry(params1_1_Kitaev)
    params = merge(params1_cry,params1_3_JDH,paramsh)
    params_Kitaev = merge(params1_1_Kitaev,params1_3_JDH,paramsh)
 
    @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" gsdata
    lsSz[i] = [gsdata["Sz"][(i,)] for i in 1:size(Latt)] |> x -> sum(x) / length(x)
end

fig = Figure()
figsize = (width = 300,height = 150)

ax = Axis(fig[1,1];figsize...,
xlabel = L"H/J_1",
ylabel = L"\langle \mathbf{S}\cdot \mathbf{H} \rangle",
xticks = 0:0.4:3,
yticks = (0:1/6:1/2,[L"0",L"\frac{1}{3}",L"\frac{2}{3}",L"1"]))

scatterlines!(ax,lsH,lsSz)

resize_to_layout!(fig)
display(fig)

save("$(figurename)/magnetization_$(Lx)x$(Ly)_$(D)_$(merge(params1_1_Kitaev,params1_3_JDH))_h=$(paramsh.h).png",fig)
save("$(figurename)/magnetization_$(Lx)x$(Ly)_$(D)_$(merge(params1_1_Kitaev,params1_3_JDH))_h=$(paramsh.h).pdf",fig)

