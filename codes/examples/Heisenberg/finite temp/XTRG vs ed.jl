using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/XTRG"
edname = "examples/Heisenberg/data/ed"
tailname = "_1"
Lx = 8
Ly = 1
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
L = size(Latt)

D = 64
DS = 2^4

params = (J = 1.0, Δ = 1.0, Hz = 0.0)

@load "$(edname)/lsdata_$(Lx)x$(Ly)_$(params).jld2" lsdata
lsdataed = lsdata 
lsIed = map(x -> real(sum(x["I"]) / L ),lsdata)

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ
# # lsβ2 = 2lsβ[2:end]
# lsI2 = map(lsI) do I
#     # @show keys(I)
#     tmp = 0.0
#     ks = keys(I)
#     for ky in ks
#         tmp += sum([I[ky][k] for k in keys(I[ky])]) / size(Latt)
#     end
#     tmp
# end .* lsβ2 

@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsF
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsE

finddata(dicts::Vector,name::String) = map(x -> x[name],dicts)
lsEed = real.(finddata(lsdataed,"E"))
# lsE = finddata(lsdata,"E")

lsFed = real.(finddata(lsdataed,"F"))
# lsF = finddata(lsdata,"F")

(lsFed .- lsF) ./ lsFed
(lsEed .- lsE) ./ lsEed