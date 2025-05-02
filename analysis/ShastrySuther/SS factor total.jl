

using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("model.jl")
Ly = 1
tailname = "SU2"
totalname = "examples/ShastrySuther/data"

# lsλ = [1.,]
lsλ = [0.0,0.2,0.4,0.65,0.66,0.7,0.75,1.0]
positions = [(i,j) for j in 1:4,i in 1:2][:]
Lx = 4
Ly = 4

@load "../codes/$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2 ^ 9

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

fig = Figure()

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


    ax = Axis(fig[positions[iλ]...];figsize...,
    title = "J₁/J₂ = $(λ)",
    ylabel = L"k_y",
    xlabel = L"k_x",
    xticks = (-2:1:2,[L"-2\pi",L"-\pi",L"0",L"\pi",L"2\pi"]),
    yticks = (-2:1:2,[L"-2\pi",L"-\pi",L"0",L"\pi",L"2\pi"]),
    )
    
    hm = heatmap!(ax,kx / pi,ky / pi,FTSS[:];colorrange = (0,iλ ≤ 4 ? 1.5 : 6.75),colormap = :hot)

    positions[iλ][1] != 2 && hidexdecorations!(ax,ticks = false)
    positions[iλ][2] != 1 && hideydecorations!(ax,ticks = false)
    iλ == 4 && Colorbar(fig[1,5],hm,label = L"\langle \mathbf{S}(\mathbf{k}) \cdot \mathbf{S}(-\mathbf{k})\rangle")
    iλ == 8 && Colorbar(fig[2,5],hm,label = L"\langle \mathbf{S}(\mathbf{k}) \cdot \mathbf{S}(-\mathbf{k})\rangle")
end

# @save "../codes/$(totalname)/AFMOD_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" AFMOD

resize_to_layout!(fig)
display(fig)

save("ShastrySuther/figures/SS factor total_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).png",fig)
save("ShastrySuther/figures/SS factor total_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).pdf",fig)




