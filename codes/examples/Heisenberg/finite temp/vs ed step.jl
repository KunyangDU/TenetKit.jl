using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/trivial"
edname = "examples/Heisenberg/data/ed"

Lx = 8
Ly = 1
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
L = size(Latt)

D = 65
DS = 2^4
τ = 0.5
Nhot = -20
βmax = 10

params = (J = 1.0, Δ = 1.0, Hz = 1.0)

@load "$(edname)/lsdata_$(Lx)x$(Ly)_$(params).jld2" lsdata
lsdataed = lsdata 
lsIed = map(x -> real(sum(x["I"]) / L ),lsdata)

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
lsβ2 = 2lsβ[2:end]
lsE = zeros(length(lsβ2))
lsF = zeros(length(lsβ2))
lsI = zeros(length(lsβ2))

for i in eachindex(lsβ2)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i+1).jld2" data
    lsE[i] = data["E"]
    lsF[i] = data["F"]
    I = data["I"]
    # @show keys(I)
    ks = keys(I)
    for ky in ks
        lsI[i] += sum([I[ky][k] for k in keys(I[ky])]) / size(Latt) * lsβ2[i]
    end
end


finddata(dicts::Vector,name::String) = map(x -> x[name],dicts)
lsEed = real.(finddata(lsdataed,"E"))
# lsE = finddata(lsdata,"E")

lsFed = real.(finddata(lsdataed,"F"))
# lsF = finddata(lsdata,"F")

lsEed .- lsE
lsIed .- lsI
