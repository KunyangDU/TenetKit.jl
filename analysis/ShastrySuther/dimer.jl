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

nnpair = neighbor(Latt)
select_points = 17:48
dimerpairs, crosspairs, nullpairs = let 
    pairs = neighbor(Latt;level=2)
    points1 = filter!(x -> Latt[x][1] ∈ [1,4] && x ∈ select_points, collect(1:size(Latt)))
    points2 = filter!(x -> Latt[x][1] ∈ [2,3] && x ∈ select_points, collect(1:size(Latt)))
    dimerpairx = filter(x -> x[1] ∈ points1 && x[2] ∈ points1 && isequal(abs.(Latt[x[1]][2].-Latt[x[2]][2]),(1,0)) , pairs)
    dimerpairy = filter(x -> x[1] ∈ points2 && x[2] ∈ points2 && isequal(abs.(mod.(Latt[x[1]][2].-Latt[x[2]][2] .+ 1, Ly) .- 1), (0,1)) , pairs)
    crosspairx = filter(x -> x[1] ∈ points2 && x[2] ∈ points2 && isequal(abs.(Latt[x[1]][2].-Latt[x[2]][2]),(1,0)) , pairs)
    crosspairy = filter(x -> x[1] ∈ points1 && x[2] ∈ points1 && isequal(abs.(mod.(Latt[x[1]][2].-Latt[x[2]][2] .+ 1, Ly) .- 1), (0,1)) , pairs)
    nullpair14 = filter(x -> x[1] ∈ points1 && x[2] ∈ points1 && isequal(abs.(mod.(Latt[x[1]][2].-Latt[x[2]][2] .+ 1, Ly) .- 1), (1,1)) , pairs)
    nullpair23 = filter(x -> x[1] ∈ points2 && x[2] ∈ points2 && isequal(abs.(mod.(Latt[x[1]][2].-Latt[x[2]][2] .+ 1, Ly) .- 1), (1,1)) , pairs)
    vcat(dimerpairx,dimerpairy),vcat(crosspairx,crosspairy),vcat(nullpair14,nullpair23)
end
dimerOP = zeros(length(lsλ))
crossOP = zeros(length(lsλ))
nullOP = zeros(length(lsλ))
for (iλ,λ) in enumerate(lsλ)
    params = (J1 = λ, J2 = 1)
    @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    dimerOP[iλ] = sum([data["SS"][pair] for pair in dimerpairs]) / length(dimerpairs)
    crossOP[iλ] = sum([data["SS"][pair] for pair in crosspairs]) / length(crosspairs)
    nullOP[iλ] = sum([data["SS"][pair] for pair in nullpairs]) / length(nullpairs)
end
@save "../codes/$(totalname)/dimerOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" dimerOP
@save "../codes/$(totalname)/crossOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" crossOP
@save "../codes/$(totalname)/nullOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" nullOP



