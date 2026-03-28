using JLD2, CairoMakie, FiniteLattices, ColorSchemes, LsqFit
include("model.jl")
Ly = 1
tailname = "SU2"
model(x,p) = @.p[1]*x^(-p[2]) + p[3]*exp(-p[4]*x)
# model(x,p) = @. p[1]*x^(p[2])
p0 = [-0.5,0.5,0.,0.]
lsλ = 0.4:0.02:0.48
lsLx = [20,]
Lx = 20
@load "../codes/examples/J1J2chain/data/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

figsize = (width = 500,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xticks = 2:2:size(Latt),
xscale = log10,
yscale = log10
)

for (i,λ) in enumerate(lsλ)
    params = (J1 = 1, J2 = λ)
    @load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-6
    bonds = zeros(N-5)
    for j in 1:N-5
        bonds[j] = data["SS"][(1,j+2)]
    end
    bonds = abs.(bonds)
    f = curve_fit(model,3:2:N-4,bonds[2:2:end],p0)
    @show norm(f.resid ./ bonds[2:2:end]),f.param
    lines!(ax,2:N-4,model(2:N-4,f.param),color = :green,linestyle = :dash)
    scatterlines!(ax,2:N-4,bonds,color = (:red))
end
resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/SSdecay-p11_$(Lx)x$(Ly).pdf",fig)
save("J1J2chain/figures/SSdecay-p11_$(Lx)x$(Ly).png",fig)






