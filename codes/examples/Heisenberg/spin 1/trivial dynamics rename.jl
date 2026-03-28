using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")

gsdataname = "examples/Heisenberg/spin 1/data/trivial"
dataname = "examples/Heisenberg/spin 1/data/trivial"

D = 3^4
Lx = 64
Ly = 1
params = (J = 1,)

@load "$(gsdataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(gsdataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(gsdataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

H = TrivialHamiltonian(Latt;params...)

v₀ = [0,0] * pi
v₁ = [1,0] * pi
lsc = vcat([i .+ (0.02:0.02:0.08) for i in 0.4:0.1:0.8]...)
lsk = [(v₀ + c*(v₁-v₀)) for c in lsc]

t₀ = 0.0
τ = 0.5
Nt = 10

xyznames = ["x","y","z"]
symbs = [:Sx,:Sy,:Sz]
ψsymbs = [:ψx,:ψy,:ψz]
 

for k in lsk
    @show k/pi
    for iψ in 1:3
        @load "$(dataname)/ψ′_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(t₀ + Nt*τ).jld2" ψ′
        @load "$(dataname)/lsSS_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(t₀)_$(t₀ + Nt*τ).jld2" lsSS

        @save "$(dataname)/ψ′_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(t₀ + Nt*τ).jld2" ψ′
        @save "$(dataname)/lsSS_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(t₀)_$(t₀ + Nt*τ).jld2" lsSS
    end
    GC.gc()
end