using CairoMakie,JLD2


dataname = "../codes/examples/Heisenberg/data/trivial"

D = 2^6

Lx = 64
Ly = 1

params = (J=1.0 , Δ = 1.0, Hz = 0.9)
τ = 0.2
Nhot = -20
Tmax = 10

Nmax = 66

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

# params = (J=1.0, Δ = 1.0, Hz = 1.0)
# lsβ = vcat(2. .^ (-15:1:-1), 1:10)
lsβ = vcat((1.0 + τ) .^ (Nhot:1:-1), 1:τ:Tmax)
lsβ2 = lsβ[2:end]*2
Is = zeros(Nmax-1)

for i in 2:Nmax
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    Is[i-1] = sum(map(x -> sum(values(x)),values(data["I"])))
end


# @load "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
# @load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
# @load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

# Is = [sum(map(x -> sum(values(x)),values(I))) for I in lsI]

figsize = (width = 400, height = 200)

fig = Figure()
ax = Axis(fig[1,1];
figsize...,
xscale = log10,
yscale = log10
)

scatterlines!(ax,1 ./ lsβ2[1:Nmax-1],abs.(Is))
xlims!(ax,10.0 ^ (-2), 10.0 ^ (1))
# xlims!(ax,1e-8,0.5)

resize_to_layout!(fig)
display(fig)

