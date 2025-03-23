using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

# λ = 0.5

Ly = 1
tailname = "SU2"

# lsλ = vcat(0:0.1:0.3, 0.4:0.01:0.6, 0.7:0.1:1)
# lsλ = vcat(0:0.1:0.3)
lsλ = 1.2:0.2:2
lsLx = [20,]
D = 2 ^ 9


for Lx in lsLx
    @show Lx
    N = Lx*Ly
    @load "examples/J1J2chain/data/Latt_$(Lx)x$(Ly).jld2" Latt

    Obs = let 
        tmp = MPSObservable()
        LocalSpace = SU₂Spin
        for i in 1:size(Latt),j in i+1:size(Latt)
            addObs!(tmp,LocalSpace.SS,(i,j),("S","S"),nothing)
        end
        tmp
    end 

    for λ in lsλ
        @show λ
        params = (J1 = 1, J2 = λ)
        tmpobs = deepcopy(Obs)
        @load "examples/J1J2chain/data/ψ_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" ψ
        @load "examples/J1J2chain/data/lsE_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" lsEg

        calObs!(tmpobs,ψ)
        data = convert(Dict{String,Any},tmpobs.values )
        data["E"] = lsEg[end]
        data["σE"] = std(lsEg[3:end])
        @save "examples/J1J2chain/data/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    end
end











