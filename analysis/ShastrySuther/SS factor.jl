using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("model.jl")
Ly = 1
tailname = "SU2"
totalname = "examples/ShastrySuther/data"

# lsλ = [1.,]
lsλ = vcat(0:0.05:0.6,0.61:0.01:0.8,0.85:0.05:1)
Lx = 4
Ly = 4

@load "../codes/$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2 ^ 9

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

figsize = (width = 250,height = 250)

lskx = range(-2pi,2pi,40)
lsky = range(-2pi,2pi,40)
lsk = [(kx,ky) for kx in lskx,ky in lsky][:]
kx,ky = eachcol(hcat(collect.(lsk)...)')
AFMOD = zeros(length(lsλ))
for (iλ,λ) in enumerate(lsλ)
    params = (J1 = λ, J2 = 1)
    @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    SS = zeros(Float64, size(Latt), size(Latt))
    for j in 1:size(Latt),i in 1:size(Latt)
        pair = i < j ? (i,j) : (j,i)
        SS[i,j] = data["SS"][pair]
    end
    FTSS = FT2(SS,Latt,lsk)

    fig = Figure()
    ax = Axis(fig[1,1];figsize...,
    title = "Static spin structure factor\n$(Lx)x$(Ly),  D = $(D),  J₁/J₂ = $(λ)",
    ylabel = L"k_y",
    xlabel = L"k_x",
    xticks = (-2:1:2,[L"-2\pi",L"-\pi",L"0",L"\pi",L"2\pi"]),
    yticks = (-2:1:2,[L"-2\pi",L"-\pi",L"0",L"\pi",L"2\pi"]),
    )

    hm = heatmap!(ax,kx / pi,ky / pi,FTSS[:];colorrange = (0,maximum(FTSS)))
    Colorbar(fig[1,2],hm,
    # label = L"F(k)",
    label = L"\langle \mathbf{S}(\mathbf{k}) \cdot \mathbf{S}(-\mathbf{k})\rangle",
    )

    # lines!(ax,ones(2) .- 1e-2,collect(extrema(lsλ)),color = :red,linestyle = :dash, linewidth = 2)
    # lines!(ax,ones(2) ./ 2,collect(extrema(lsλ)),color = :yellow,linestyle = :dash, linewidth = 2)

    ind = findfirst(x -> x[1] ≈ 2pi && x[2] ≈ 2pi,lsk)
    AFMOD[iλ] =  FTSS[ind]

    resize_to_layout!(fig)
    display(fig)

    save("ShastrySuther/figures/SS factor/SS factor_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).png",fig)
    save("ShastrySuther/figures/SS factor/SS factor_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).pdf",fig)
end

@save "../codes/$(totalname)/AFMOD_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" AFMOD






