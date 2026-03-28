using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LsqFit
include("../../analysis/analysis.jl")
include("model.jl")

dataname = "../codes/examples/Heisenberg/spin 1/data/trivial"

D = 3^4
Ly = 1
params = (J=1,)

lsLx = vcat(100:20:200,240:40:400)
lsE = zeros(length(lsLx))
for (i,Lx) in enumerate(lsLx) 
    @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
    lsE[i] = lsEg[end] /Lx /Ly
end

E0 = -1.401484039
fig = Figure()
ax = Axis(fig[1,1])
model(x,p) = @. x*p[1] + p[2]
f = curve_fit(model,1 ./ lsLx,lsE,zeros(2))
lines!(ax,0:0.005:0.01,model(0:0.005:0.01,f.param))
scatter!(ax,1 ./ lsLx, lsE)
scatter!(ax,0,E0)

xlims!(ax,0,0.01)

resize_to_layout!(fig)
display(fig)

model(0,f.param) - E0

