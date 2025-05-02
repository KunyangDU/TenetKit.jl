using JLD2, CairoMakie, FiniteLattices, ColorSchemes
include("../analysis/analysis.jl")
include("model.jl")
Ly = 1
tailname = "SU2"
totalname = "examples/ShastrySuther/data"

# lsλ = 0:0.05:1
# lsλ = [0.,]
lsλ = vcat(0:0.05:0.6,0.61:0.01:0.8,0.85:0.05:1)
Lx = 4
Ly = 4

@load "../codes/$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 2 ^ 9
select_points = 17:48
plaquettepairs,nullpairs = let 
    pairs = neighbor(Latt;level=1)
    points1 = filter!(x -> Latt[x][1] ∈ [1,4] && x ∈ select_points, collect(1:size(Latt)))
    points2 = filter!(x -> Latt[x][1] ∈ [2,3] && x ∈ select_points, collect(1:size(Latt)))
    plaquettepairs = filter(x -> (x[1] ∈ points1) ⊻ (x[2] ∈ points1) && isequal(abs.(Latt[x[1]][2].-Latt[x[2]][2]),(0,0)) , pairs)
    nullpairs = filter(x -> (x[1] ∈ points1) ⊻ (x[2] ∈ points1) && !isequal(abs.(Latt[x[1]][2].-Latt[x[2]][2]),(0,0)) , pairs)
    plaquettepairs,nullpairs
end
plaquetteOP = zeros(length(lsλ))
for (iλ,λ) in enumerate(lsλ)
    params = (J1 = λ, J2 = 1)
    @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    plaquetteOP[iλ] = sum([data["SS"][pair] for pair in plaquettepairs]) / length(plaquettepairs) - sum([data["SS"][pair] for pair in nullpairs]) / length(nullpairs)
end
@save "../codes/$(totalname)/plaquetteOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" plaquetteOP



