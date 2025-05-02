using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../analysis/analysis.jl")
include("model.jl")
Ly = 1
tailname = "SU2"
totalname = "examples/ShastrySuther/data"

# lsλ = 0:0.05:1
# lsλ = [0.,]
lsλ = 0.6:0.01:0.8
Lx = 4
Ly = 4

@load "../codes/$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2 ^ 9

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

figsize = (width = 75Lx,height = 75Ly)

for (i,λ) in enumerate(lsλ)
    params = (J1 = λ, J2 = 1)
    @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    @assert abs(data["σE"] / data["E"]) < 1e-1 data["σE"] / data["E"]
    @show λ,data["σE"]
    nb = neighbor(Latt;level = 1)
    pairx,pairy = _ShastrySutherPairs(Latt)
    bonds1,bondsx,bondsy = map(y -> map(x -> data["SS"][x],y),[nb,pairx,pairy])
    # bonds /= maximum(abs.(bonds))

    fig = Figure()
    ax = Axis(fig[1,1];figsize...,
    title = "⟨Sᵢ⋅Sⱼ⟩ pattern\n$(Lx)x$(Ly),  D = $(D),  J₁/J₂ = $(λ)",
    xlabel = L"L_x",
    ylabel = L"L_y",
    autolimitaspect = true,
    # xticks = 1:1:size(Latt) |> x -> (x,string.(x)),
    # yticks = 1:1:length(lsλ) |> x -> (x,string.(lsλ)),
    xticks = (1:0.5:Lx+0.5,string.(1:2Lx)),yticks = (1:0.5:Ly+0.5,string.(1:2Ly))
    )

    plotLatt!(ax,Latt,[0,1];site = true,tplevel=1,sitelabel = false,sitesize = 12*ones(size(Latt)))
    plotLatt!(ax,Latt,[0,1];site = false,pairs = pairx)
    plotLatt!(ax,Latt,[0,1];site = false,pairs = pairy)
    clim = 0.75
    λ < 0.66 ? (clim = 0.75) : (clim = (log(2)-1/4)*λ)

    plotbond!(ax,Latt,nb,bonds1,[0,1];colorlimit = clim .* (-1,1),colormap = :bwr)
    plotbond!(ax,Latt,pairx,bondsx,[0,1];colorlimit = clim .* (-1,1),colormap = :bwr)
    plotbond!(ax,Latt,pairy,bondsy,[0,1];colorlimit = clim .* (-1,1),colormap = :bwr)
    Colorbar(fig[1,2],limits = clim .* (-1,1),colormap = :bwr, label = L"\langle \mathbf{S}_i\cdot\mathbf{S}_j \rangle")
    # text!(ax,0,i + 0.5;text = "$(λ)",fontsize = 12)
    resize_to_layout!(fig)
    display(fig)
    save("ShastrySuther/figures/SS pattern/SS pattern_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).pdf",fig)
    save("ShastrySuther/figures/SS pattern/SS pattern_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).png",fig)
end







