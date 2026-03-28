using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/lab/data"


D = 20
Lx = 8
Ly = 1
params = (J = 1, Δ = 1, Hx = 0,Hz = 0, hx = 0.0)

Latt = YCSqua(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
@load "$(dataname)/H_$(Lx)x$(Ly)_$(D)_$(params).jld2" H

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinOneHalf.PhySpace ,AuxSpace)
end

t₀ = 0.0
τ = 0.5
Nt = 50

info = TDVPinfo()
alg = TDVPalgo(DoubleSite(),
    NoAlgorithm(),
    truncdim(D) & truncbelow(1e-12),τ/2,1,TDVPDefaultLanczos,true,false
)
@time "initialize environment" begin 
    Env = Environment([ψ,H,ψ'])
    initialize!(Env)
end
flush(stdout)

for i in 1:Nt
    t = t₀ + τ * i
    println("t = $(t)")
    flush(stdout)
    
    TDVP!(Env, alg, info)
    normalize!(ψ)
    @show _scalar(Env)
end


