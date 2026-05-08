using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/trivial"
edname = "examples/Heisenberg/data/ed"

Lx = 2
Ly = 4
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
L = size(Latt)

D = 256
DS = 2^4
τ = 0.5
Nhot = -20
βmax = 10

params = (J = 1.0, Δ = 1.0, Hz = 1.0)

@load "$(edname)/lsdata_$(Lx)x$(Ly)_$(params).jld2" lsdata
lsdataed = lsdata 
lsIed = map(x -> real(sum(x["I"]) / L ),lsdata)

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
lsβ2 = 2lsβ[2:end]
lsI2 = map(lsI) do I
    # @show keys(I)
    tmp = 0.0
    ks = keys(I)
    for ky in ks
        tmp += sum([I[ky][k] for k in keys(I[ky])]) / size(Latt)
    end
    tmp
end .* lsβ2 

@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsF
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsE
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ


finddata(dicts::Vector,name::String) = map(x -> x[name],dicts)
lsEed = real.(finddata(lsdataed,"E"))
# lsE = finddata(lsdata,"E")

lsFed = real.(finddata(lsdataed,"F"))
# lsF = finddata(lsdata,"F")

lsEed .- lsE
# ),norm(lsFed .- lsF)

(lsI2 .- lsIed) ./ lsIed
tr.(lsρ)

# lsIed
