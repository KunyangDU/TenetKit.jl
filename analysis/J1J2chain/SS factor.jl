using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("model.jl")
Ly = 1
tailname = "SU2"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
# lsλ = 0:0.1:1
# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1,1.2:0.2,2)
lsλ = 0:0.1:2
lsLx = [20,]
Lx = 20
@load "../codes/examples/J1J2chain/data/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 10
J1 = -1

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

figsize = (width = 400,height = 300)


lsk = range(-pi,pi,53)

F = zeros(ComplexF64,length(lsλ),length(lsk))
for (iλ,λ) in enumerate(lsλ)
    params = (J1 = J1, J2 = λ)
    @load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    F[iλ,:] = let 
        tmp = zeros(ComplexF64,length(lsk))
        for i in 1:size(Latt), j in 1:size(Latt)
            abs(i - j) ∉ 1:6 && continue
            SS = i<j ? data["SS"][(i,j)] : data["SS"][(j,i)]
            # if λ == 0 && SS < 0
            #     @show (i,j),SS
            # end
            for ik in eachindex(lsk)
                tmp[ik] += SS * exp(1im * lsk[ik] * (coordinate(Latt,i) .- coordinate(Latt,j))[1])/size(Latt)
            end
        end
        tmp .+= 3/4
        tmp
    end
end


F = abs.(F)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
ylabel = L"J_2/J_1",
xlabel = L"k_x",
xticks = (-1:0.5:1,[L"-\pi",L"-\pi/2",L"0",L"\pi/2",L"\pi"]),
yticks = 0:0.5:2)

# contourf!(ax,lsk ./ pi,lsλ,F',levels = 50)
# Colorbar(fig[1,2],limits = extrema(F),
hm = heatmap!(ax,lsk ./ pi,lsλ,F')
Colorbar(fig[1,2],hm,
# label = L"\frac{1}{N}\sum_{i,j}\langle \mathbb{S}_i \cdot \mathbb{S}_j\rangle\mathrm{e}^{ik\cdot(R_i - R_{j})}")
label = L"|F(k)|")

lines!(ax,ones(2) .- 1e-2,collect(extrema(lsλ)),color = :red,linestyle = :dash, linewidth = 2)
lines!(ax,ones(2) ./ 2,collect(extrema(lsλ)),color = :yellow,linestyle = :dash, linewidth = 2)

resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/SS factor_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).pdf",fig)
save("J1J2chain/figures/SS factor_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).png",fig)

nnafm = F[:,findfirst(x -> x ≈ pi/2,lsk)]
nnnafm = F[:,findfirst(x -> x ≈ pi,lsk)]

@save "J1J2chain/data/NNAFM_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nnafm
@save "J1J2chain/data/NNNAFM_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nnnafm


