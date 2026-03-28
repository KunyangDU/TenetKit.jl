using JLD2, CairoMakie, FiniteLattices, ColorSchemes, LsqFit
include("../model.jl")
Ly = 1
tailname = "SU2"
modele(x,p) = @. p[1]*exp(-p[2]*x)/x^(-p[3])
modela(x,p) = @. p[1]*x^(-p[2])
lsλ = 0:0.05:0.5
lsLx = [20,]
Lx = 22
@load "../codes/examples/J1J2chain/data/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9
J1 = -1

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

figsize = (width = 500,height = 200)

nncoeff = []

for (i,λ) in enumerate(lsλ)
    fig = Figure()
    ax = Axis(fig[1,1];figsize...,
    xticks = 2:2:size(Latt),
    xscale = log10,
    # yscale = log10,
    title = "λ = $(λ)"
    )
    params = (J1 = J1, J2 = λ)
    @load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-1 λ
    bonds = zeros(N-1)
    for j in 1:N-1
        bonds[j] = data["SS"][(1,j+1)]
    end
    site = 1:N-1
    selected_site = 3:2:N-4
    selected_bonds = bonds[selected_site]
    inds = findall(x -> selected_bonds[1] * x < 0,selected_bonds)
    @show inds
    # !isempty(inds) && continue
    # bonds = abs.(bonds)
    selected_bonds = abs.(selected_bonds)
    cl = let 
        # fe = curve_fit(modele,selected_site,selected_bonds,[0.5,0.,0.5])
        fa = curve_fit(modela,selected_site,selected_bonds,[0.5,0.5])
        # ϵ = [norm(f.resid ./ selected_bonds) for f in [fe,fa]]
        # @show min(ϵ...)
        # lines!(ax,selected_site,modele(selected_site,fe.param),color = :purple,linestyle = :dash)
        # lines!(ax,selected_site,modela(selected_site,fa.param),color = :blue,linestyle = :dash)
        # ϵ[1] < ϵ[2] ? 1/fe.param[2] : fa.param[2]
        fa.param[2]
    end
    push!(nncoeff,[λ,cl])

    scatterlines!(ax,site,bonds,color = (:red))
    # scatter!(ax,selected_site,selected_bonds,color = (:blue))
    resize_to_layout!(fig)
    display(fig)
end


# save("J1J2chain/figures/SSdecay-p2_$(Lx)x$(Ly).pdf",fig)
# save("J1J2chain/figures/SSdecay-p2_$(Lx)x$(Ly).png",fig)

nncoeff = hcat(nncoeff...)

@save "J1J2chain/data/NNcoefficient_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nncoeff





