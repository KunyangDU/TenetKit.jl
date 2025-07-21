using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes
include("../../analysis/analysis.jl")
include("../model.jl")

trivialname = "../codes/examples/Heisenberg/data/trivial"
SU2name = "../codes/examples/Heisenberg/data/SU2"
U1name = "../codes/examples/Heisenberg/data/U1"
dataname = "Heisenberg/data"
D = 2^7
Lx = 14
Ly = 1
# Latt = YCSqua(Lx,Ly)
@load "$(trivialname)/Latt_$(Lx)x$(Ly).jld2" Latt

trparams = (J=1,)
u1params = (Jz = 1,Jxy = 0.5,h=0)
su2params = (J=1,)
edparams = (Jz=1,Jxy=1)
lsβed = vcat(2. .^ (-5:1:-1), 1:10) .* 2
@load "$(trivialname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(trparams).jld2" lsβ2

# @load "$(trivialname)/data_$(Lx)x$(Ly)_$(2D)_$(trparams).jld2" data
# datatr = data
# Ctr = @. real((datatr["E2"] - datatr["E"]^2) * lsβ2 ^ 2)
# χtr = [(calcFDT(Latt,datatr["obs"][i]["SzSz"],datatr["obs"][i]["Sz2"]) - sum([datatr["obs"][i]["Sz"][(j,)] for j in 1:size(Latt)])^2) for i in eachindex(lsβ2)] .* lsβ2
# lsEtr = datatr["E"]
# lsFtr = datatr["F"]
Ctr = zeros(length(lsβ2))
χtr = zeros(length(lsβ2))
lsEtr = zeros(length(lsβ2))
lsFtr = zeros(length(lsβ2))

for (i,β) in enumerate(lsβ2)
    i == 1 && continue
    @load "$(trivialname)/data_$(Lx)x$(Ly)_$(D)_$(trparams)_$(i+1).jld2" data
    lsEtr[i] = data["E"]
    lsFtr[i] = data["F"]
    # χtr[i-1] = data[]
    # Ctr[i-1] = real((data["E2"] - data["E"]^2) * β ^ 2)
end


@load "$(trivialname)/lsE_$(Lx)x$(Ly)_$(D)_$(trparams).jld2" lsE
lsEtr = lsE
@load "$(SU2name)/lsE_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" lsE
lsEsu2 = lsE
@load "$(U1name)/lsE_$(Lx)x$(Ly)_$(D)_$(u1params).jld2" lsE
lsEu1 = lsE
# @load "$(U1name)/data_$(Lx)x$(Ly)_$(D)_$(u1params).jld2" data
# datau1 = data
# Cu1 = @. real((datau1["E2"] - datau1["E"]^2) * lsβ2 ^ 2)
# χu1 = [(calcFDT(Latt,datatr["obs"][i]["SzSz"],datatr["obs"][i]["Sz2"]) - sum([datatr["obs"][i]["Sz"][(j,)] for j in 1:size(Latt)])^2) for i in eachindex(lsβ2)] .* lsβ2
# lsEu1 = datau1["E"]
# lsFu1 = datau1["F"]

# @load "$(SU2name)/data_$(Lx)x$(Ly)_$(D)_$(su2params).jld2" data
# datasu2 = data
# Csu2 = @. real((datasu2["E2"] - datasu2["E"]^2) * lsβ2 ^ 2)
# χsu2 = [calcFDT(Latt,datasu2["obs"][i]["SS"]) + 3size(Latt)/4 for i in eachindex(lsβ2)] .* lsβ2 
# lsEsu2 = datasu2["E"]
# lsFsu2 = datasu2["F"]

@load "$(dataname)/data_obc_$(Ly)x$(Lx)_$(edparams).jld2" data
dataed = data
χed = dataed["χ"]
Ced = dataed["C"]
Eed = dataed["E"]


# (lsEsu2[end-length(Eed)+1:end] .- Eed) ./ Eed
# Csu2[end-length(Eed)+1:end] .- Ced
# χsu2[end-length(Eed)+1:end] .- χed
# (lsE[end-length(Eed)+1:end] .- Eed) ./ Eed
# (lsEtr[end-length(Eed)+1:end] .- Eed) ./ Eed
# (lsEsu2[end-length(Eed)+1:end] .- Eed) ./ Eed
(lsEu1[end-length(Eed)+1:end] .- Eed) ./ Eed

# Ctr[end-length(Eed)+1:end] .- Ced
# lsEtr .- lsEsu2
# lsEtr .- lsEu1

# lsFtr .- lsFsu2
# Eed

