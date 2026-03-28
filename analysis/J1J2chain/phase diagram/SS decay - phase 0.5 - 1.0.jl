using JLD2, CairoMakie, FiniteLattices, ColorSchemes, LsqFit
include("model.jl")
Ly = 1
tailname = "SU2"
# model(x,p) = @.p[1]*x^(-p[2]) + p[3]*exp(-p[4]*x)
model(x,p) = @. p[1]*x^(p[2])
p0 = [0.5,-0.5,0.,0.]
lsλ = 0.6:0.1:1.4
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
    bonds = zeros(N-1)
    for j in 1:N-1
        bonds[j] = data["SS"][(1,j+1)]
    end
    bonds = abs.(bonds)
    site = 1:N-1
    selected_site = 2:4:N-1
    selected_bonds = bonds[selected_site]
    f = curve_fit(model,selected_site,selected_bonds,p0)
    @show norm(f.resid ./ selected_bonds),f.param
    lines!(ax,site,model(site,f.param),color = (:green,i/length(lsλ)),linestyle = :dash)
    scatterlines!(ax,site,bonds,color = (:red,i/length(lsλ)))
    scatter!(ax,selected_site,selected_bonds,color = (:blue,i/length(lsλ)))
end
resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/SSdecay-p2_$(Lx)x$(Ly).pdf",fig)
save("J1J2chain/figures/SSdecay-p2_$(Lx)x$(Ly).png",fig)






