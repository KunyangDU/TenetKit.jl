using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/YuanYao/data"

D = 256
# Lx = 20
Ly = 1
params = (J₁ = 1.0, J₂ = 0.0)
PBC = true
for Lx in 40:10:120

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params)_$(PBC).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params)_$(PBC).jld2" ψ

n = [0.0, 0.0, 1.0]

begin
    Obs = Observable()
    LocalSpace = TrivialSpinOneHalf

    S = [LocalSpace.Sx,LocalSpace.Sy,LocalSpace.Sz]
    Sn = sum(n .* S)

    sites = collect(1:size(Latt))
    addString!(Obs,[exp(4pi*1im*i*Sn/size(Latt)) for i in sites],sites,["Sn" for _ in sites],[false for _ in sites],nothing)
    # for L in 100:50:size(Latt)
    #     sites = collect(div(size(Latt)-L,2)+1:L+div(size(Latt)-L,2))
    #     addString!(Obs,[exp(4pi*1im*i*Sn/L) for i in sites],sites,["Sn" for _ in sites],[false for _ in sites],nothing)
    # end
    calObs!(Obs, ψ)
end

gsdata = Obs.values
@save "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params)_$(PBC).jld2" gsdata
end

# gsdata[(Tuple(["Sn" for i in 1:size(Latt)]),)][(Tuple([i for i in 1:size(Latt)]),)]



