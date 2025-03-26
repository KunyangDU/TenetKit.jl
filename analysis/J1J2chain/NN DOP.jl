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
J1 = 1

nnpair = neighbor(Latt)
lsE = zeros(length(lsλ))

figsize = (width = 400,height = 300)


lsk = range(-pi,pi,53)

nndop = zeros(ComplexF64,length(lsλ))
for (iλ,λ) in enumerate(lsλ)
    params = (J1 = J1, J2 = λ)
    @load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    nndop[iλ] = let 
        tmp = 0
        for i in 1:size(Latt)-2
            tmp += (-1)^i * (data["SS"][(i,i+1)] - data["SS"][(i+1,i+2)]) / sqrt(size(Latt))
        end
        tmp
    end
end


nndop = abs.(nndop)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
xlabel = L"J_2/J_1",
ylabel = L"DOP")


# lines!(ax,ones(2) .- 1e-2,collect(extrema(lsλ)),color = :red,linestyle = :dash, linewidth = 2)
# lines!(ax,ones(2) ./ 2,collect(extrema(lsλ)),color = :yellow,linestyle = :dash, linewidth = 2)

scatterlines!(ax,lsλ,nndop)

resize_to_layout!(fig)
display(fig)

save("J1J2chain/figures/DOP_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).pdf",fig)
save("J1J2chain/figures/DOP_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).png",fig)

@save "J1J2chain/data/DOP_$(Lx)x$(Ly)_D=$(D)_J1=$(J1)_$(tailname).jld2" nndop


