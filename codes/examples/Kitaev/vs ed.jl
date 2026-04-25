using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")

dataname = "examples/Kitaev/data"
edname = "examples/Kitaev/data/ed"

Lx = 1
Ly = 2
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
L = size(Latt)

D = 64
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 100

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.1)

@load "$(edname)/lsdata_$(Lx)x$(Ly)_$(params).jld2" lsdata
lsdataed = lsdata 
lsIed = map(x -> real(sum(x["I"]) / L ),lsdata)

@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2
lsdata = Dict[]
lsI = zeros(length(lsβ2))
for (i,β) in enumerate(lsβ2)
    @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i + 1).jld2" data
    push!(lsdata,data)
    I = data["I"]
    ks = keys(I)
    for k in ks
        lsI[i] += real(sum([I[k][s] for s in keys(I[k])]) / size(Latt) * lsβ2[i])
    end
end

finddata(dicts::Vector,name::String) = map(x -> x[name],dicts)
lsEed = real.(finddata(lsdataed,"E"))
lsE = finddata(lsdata,"E")

lsFed = real.(finddata(lsdataed,"F"))
lsF = finddata(lsdata,"F")

lsEed .- lsE
lsFed .- lsF

lsIed .- lsI
