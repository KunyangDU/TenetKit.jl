using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

# λ = 0.5

tailname = "SU2"
totalname = "examples/ShastrySuther/data"

lsλ = vcat(0.61:0.01:0.64,0.71:0.01:0.74,0.76:0.01:0.79)
# lsλ = [0.,]
Lx = 4
Ly = 4
D = 2 ^ 9

@load "$(totalname)/Latt_$(Lx)x$(Ly).jld2" Latt

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
    params = (J1 = λ, J2 = 1)
    tmpobs = deepcopy(Obs)
    @load "$(totalname)/ψ_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" ψ
    @load "$(totalname)/lsE_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" lsEg

    calObs!(tmpobs,ψ)
    data = convert(Dict{String,Any},tmpobs.values )
    for i in 1:size(Latt)
        data["SS"][(i,i)] = 3/4
    end
    data["E"] = lsEg[end]
    data["σE"] = std(lsEg[end-2:end])
    @save "$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
end












