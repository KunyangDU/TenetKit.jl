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
pairs = filter(x -> x[1] ∈ select_points && x[2] ∈ select_points,neighbor(Latt;level=1))
NNSSOP = zeros(length(lsλ))
for (iλ,λ) in enumerate(lsλ)
    params = (J1 = λ, J2 = 1)
    @load "../codes/$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    NNSSOP[iλ] = sum([data["SS"][pair] for pair in pairs]) / length(pairs)
end
@save "../codes/$(totalname)/NNSSOP_$(Lx)x$(Ly)_D=$(D)_$(tailname).jld2" NNSSOP


