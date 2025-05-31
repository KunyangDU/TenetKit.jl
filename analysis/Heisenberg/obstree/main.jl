using CairoMakie,JLD2,TensorKit,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("model.jl")

newname = "Heisenberg/obstree/data/new"
oldname = "Heisenberg/obstree/data/old"

Lx = 4
Ly = 4
D = 2^7
params = (J = 1, H = 5)

@load "$(newname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(newname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(newname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
lsEnew = deepcopy(lsEg)
datanew = deepcopy(gsdata)

@load "$(oldname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(oldname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(oldname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
lsEold = deepcopy(lsEg)
dataold = deepcopy(gsdata)

ΔE = (lsEnew .- lsEold)[end]

lskx = pi*range(-1,1,51)
lsky = pi*range(-1,1,51)
lsk = filter(x -> isinside(x,FBZpoint;isboundary = true),[[kx,ky] for kx in lskx,ky in lsky][:])
lstk = map(x -> Tuple(x),lsk)
FSSold = TrivialSSFT(Latt,dataold,lstk)
FSSnew = TrivialSSFT(Latt,datanew,lstk)
FSStold = sum(FSSold)
FSStnew = sum(FSSnew)
Smold = (maximum(FSStold))
Smnew = (maximum(FSStnew))

maximum(abs.(FSStold .- FSStnew)),ΔE
lsEold ./ size(Latt) .- 1/4
lsEnew ./ size(Latt) .- 1/4