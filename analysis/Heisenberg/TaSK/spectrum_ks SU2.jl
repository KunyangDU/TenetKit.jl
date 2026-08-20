using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")

dataname = "../codes/examples/Heisenberg/laboratory/data/TaSK/SU2"

figurename = "Heisenberg/TaSK/figures"

D = 64
Lx = 64
Ly = 1
J = 1.0
params = (J= 1.0,)

N = 100

figsize = (width = 400,height = 300)

fig = Figure()
ax = Axis(fig[1,1];
title = "SU2, $(Lx)x$(Ly) Squa, D = $(D), $(params), N = $(N)",
figsize...,
xlabel = L"k_x/\pi",
ylabel = L"\omega")

lsω = 0:0.01:4
lsk = 0:0.04:1.0
As = zeros(length(lsk),length(lsω))

for (ik,k) in enumerate(lsk)
kv = [k,0.0]
@load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(kv)_$(N).jld2" data
inds = filter(x -> data["S"][x] > 1e-4, 1:length(data["S"]))
ω = data["ω"][inds]
S = sqrt.(data["S"][inds])
# As[ik,:] = map(x -> gaussian(x,ω,S;σ = 0.5),lsω)
As[ik,:] = cfe(data["a"],data["b"][1:end-1],data["d"],lsω;η = 0.2,K = 40)
end

hm = heatmap!(ax,lsk,lsω,As,colorrange = (0,1.5),colormap = colorschemes[:jet])

lines!(ax,lsk,@. sin(lsk * pi)/2*pi;color = :red,linestyle = :dash)
lines!(ax,lsk,@. sin(lsk * pi/2)*pi;color = :red,linestyle = :dash)

Colorbar(fig[1,2],hm,label = L"A_k(\omega)")
resize_to_layout!(fig)

display(fig)
save("$(figurename)/Akomega_SU2_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(N).png",fig)
save("$(figurename)/Akomega_SU2_$(Lx)x$(Ly)_$(D)_$(params)_$(k)_$(N).pdf",fig)




