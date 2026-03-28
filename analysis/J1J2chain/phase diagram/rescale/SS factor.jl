using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../model.jl")
Ly = 1
tailname = "SU2"
totalname = "examples/J1J2chain/data/rescale"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
# lsλ = 0:0.1:1
# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1,1.2:0.2,2)
# lsλ = 0:0.05:1
lsλp = vcat(0.5:0.5:5)
# lsλp = vcat(0.2:0.2:2,2.5:0.5:5)
lsλ = vcat(-reverse(lsλp),0,lsλp)
lsLx = [20,]
Lx = 20
@load "../codes/$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

figsize = (width = 400,height = 300)


lsk = range(-pi,pi,53)

F = zeros(ComplexF64,length(lsλ),length(lsk))
for (iλ,λ) in enumerate(lsλ)
    params = (J1 = λ, J2 = 1)
    @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    F[iλ,:] = let 
        tmp = zeros(ComplexF64,length(lsk))
        for i in 1:size(Latt), j in 1:size(Latt)
            abs(i - j) ∉ 1:6 && continue
            tmpi,tmpj = i<j ? (i,j) : (j,i)
            # if λ == 0 && SS < 0
            #     @show (i,j),SS
            # end
            for ik in eachindex(lsk)
                tmp[ik] += data["SS"][(tmpi,tmpj)] * exp(1im * lsk[ik] * (coordinate(Latt,tmpi) .- coordinate(Latt,tmpj))[1])/size(Latt)
            end
        end
        tmp .+= 3/4
        tmp
    end
end


F = abs.(F)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
ylabel = L"J_1/J_2",
xlabel = L"k_x",
xticks = (-1:0.5:1,[L"-\pi",L"-\pi/2",L"0",L"\pi/2",L"\pi"]),
yticks = -10:0.5:10)

# contourf!(ax,lsk ./ pi,lsλ,F',levels = 50)
# Colorbar(fig[1,2],limits = extrema(F),
hm = heatmap!(ax,lsk ./ pi,lsλ,F')
Colorbar(fig[1,2],hm,
# label = L"\frac{1}{N}\sum_{i,j}\langle \mathbb{S}_i \cdot \mathbb{S}_j\rangle\mathrm{e}^{ik\cdot(R_i - R_{j})}")
label = L"|F(k)|")

lines!(ax,ones(2) ,collect(extrema(lsλ)),color = :red,linestyle = :dash, linewidth = 2)
lines!(ax,ones(2) ./ 2,collect(extrema(lsλ)),color = :blue,linestyle = :dash, linewidth = 2)
lines!(ax,zeros(2),collect(extrema(lsλ)),color = :green,linestyle = :dash, linewidth = 2)
lines!(ax,[-1,1],zeros(2),color = :black, linewidth = 2)

resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/rescale/SS factor_$(Lx)x$(Ly)_D=$(D)_$(tailname).pdf",fig)
save("J1J2chain/figures/rescale/SS factor_$(Lx)x$(Ly)_D=$(D)_$(tailname).png",fig)

nnafm = F[:,findfirst(x -> x ≈ pi,lsk)]
nnnafm = F[:,findfirst(x -> x ≈ pi/2,lsk)]
nnfm = F[:,findfirst(x -> x ≈ 0,lsk)]
maxss = [maximum(F[i,:]) for i in eachindex(lsλ)]
@save "J1J2chain/data/rescale/NNAFM_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nnafm
@save "J1J2chain/data/rescale/NNNAFM_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nnnafm
@save "J1J2chain/data/rescale/NNFM_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nnfm
@save "J1J2chain/data/rescale/MAXSS_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" maxss


