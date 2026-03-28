using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

gsdataname = "examples/Heisenberg/trivial/data"
dataname = "examples/Heisenberg/trivial/data"

D = 2^6
Lx = 10
Ly = 1
params = (J = 1, Δ = 1)

@load "$(gsdataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(gsdataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(gsdataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

H = TrivialHamiltonian(Latt;params...)

v₀ = [0,0] * pi
v₁ = [1,0] * pi
lsc = 0:0.2:1
lsk = [(v₀ + c*(v₁-v₀)) for c in lsc]

t = 5
Nt = 11

for k in lsk
@show k/pi
Hs = let LocalSpace = TrivialSpinOneHalf
    map([:Sx,:Sy,:Sz]) do Sop 
        Root = InteractionTreeNode()
        for i in 1:size(Latt)
            addIntr!(Root,LocalSpace.eval(Sop),i,string(Sop),false,exp(1im * dot(k,coordinate(Latt,i))),nothing)
        end
        AutomataSparseMPO(Root,size(Latt))  
    end
end

ψs = map(Hs) do H′ 
    mul!(deepcopy(ψ),ψ,H′ ,1,Algebraalgo(DoubleSite(),NoAlgorithm(),truncdim(D) & truncbelow(1e-12),3,1e-8))[1]
end

lsSS = map(ψs) do ψ1
    lst, lsψ,lsinfo = TDVP1!(deepcopy(ψ1),H,t,Nt;trunc = truncdim(D) & truncbelow(1e-12),tol = 1)  
    map(enumerate(lsψ)) do (i,ψ′)
        inner(ψ′,ψ1') * exp(1im * lst[i] * lsEg[end])
    end
end
@save "$(dataname)/lsSS_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(t)_$(Nt).jld2" lsSS
lsSS
GC.gc()
end
