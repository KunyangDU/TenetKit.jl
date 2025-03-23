using JLD2, CairoMakie, FiniteLattices

Ly = 1
tailname = "SU2"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
lsλ = 0:0.1:1
lsLx = [20,]
Lx = 20
@load "../codes/examples/J1J2chain/data/Latt_$(Lx)x$(Ly).jld2" Latt

N = Lx*Ly
D = 2 ^ 9
lsE = zeros(length(lsλ))

λ = 0.
params = (J1 = 1, J2 = λ)
@load "../codes/examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
@assert abs(data["σE"] / data["E"]) < 1e-6
for i in 1:size(Latt)-1
    @show data["SS"][(i,i+1)]
end

for i in 1:size(Latt)-1
    @show data["SS"][(1, 1 + i)]
end










