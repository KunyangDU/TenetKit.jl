using TensorKit
include("../../../src/iMPS.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^6
Lx = 10
Ly = 1
params = (J=1,)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

H = TrivialHamiltonian(Latt;params...)

lskx = 2pi * range(0,1,11)
# lskx = [0,pi]
lsky = [0,]
lsk = [[kx,ky] for kx in lskx, ky in lsky][:]

t = 5
Nt = 10
for k in lsk
    Hs = let LocalSpace = TrivialSpinOneHalf
        map([:Sx,:Sy,:Sz]) do Sop 
            Root = InteractionTreeNode()
            for i in 1:size(Latt)
                addIntr!(Root,LocalSpace.eval(Sop),i,string(Sop),exp(1im * dot(k,coordinate(Latt,i))),nothing)
            end
            AutomataSparseMPO(InteractionTree(Root),size(Latt))  
        end
    end

    ψs = map(Hs) do H′ 
        mul!(deepcopy(ψ),ψ,H′ ,1,Algebraalgo(DoubleSite(),NoAlgorithm(),truncdim(D) & truncbelow(1e-12),3,1e-8))[1]
    end

    lsSS = map(ψs) do ψ1
        lst, lsψ,lsinfo = TDVP1!(deepcopy(ψ1),H,t,Nt;trunc = truncdim(D) & truncbelow(1e-12))  
        map(enumerate(lsψ)) do (i,ψ′)
            inner(ψ′,ψ1') * exp(1im * lst[i] * lsEg[end])
        end
    end
    @show lsSS
    @save "$(dataname)/SS/lsSS_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k;digits = 3))_$(t)_$(Nt).jld2" lsSS
end
