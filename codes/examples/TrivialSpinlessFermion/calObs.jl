using TensorKit,JLD2,KrylovKit
include("../../src/TenetKit.jl")
include("model.jl")

Lx = 6
Ly = 1
D = 2^6

@load "examples/TrivialSpinlessFermion/data/Latt_$(Lx)x$(Ly).jld2" Latt
@load "examples/TrivialSpinlessFermion/data/ψ_$(Lx)x$(Ly)_$(D).jld2" ψ



begin
    Obs = Observable()
    LocalSpace = TrivialSpinlessFermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",false,nothing)
    end
    # @show Obs.node

    # Obs.node

    calObs!(Obs, ψ)
end

gsdata = Obs.values

# gsdata[(("n",),)]




