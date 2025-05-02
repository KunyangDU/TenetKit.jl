using TensorKit
include("../../src/iMPS.jl")
include("model.jl")

# λ = 0.5
totalname = "examples/J1J2chain/data/rescale"

Ly = 1
tailname = "SU2"

lsλ = vcat(0.2:0.2:2,2.5:0.5:5,6:10)
lsλ = vcat(-reverse(lsλ),0,lsλ)
lsλ = vcat(lsλ,[-1.5,-0.5,0.5,1.5])
lsLx = [20,]
D = 2 ^ 9


for Lx in lsLx
    @show Lx
    N = Lx*Ly
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
        data["E"] = lsEg[end]
        data["σE"] = std(lsEg[end-2:end])
        @save "$(totalname)/data_$(Lx)x$(Ly)_D=$(D)_params=$(params)_$(tailname).jld2" data
    end
end











