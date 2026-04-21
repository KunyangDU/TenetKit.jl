using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^6
# for Ly in 4:2:20,Lx in Ly:2:20 
Lx = 64
Ly = 1

Latt = YCSqua(Lx,Ly)
Hz = 3.0
J = 1.0
Δ = 1.0
params = (J=J, Δ = Δ, Hz = Hz)

L = size(Latt)
lsk = collect((1:L)*pi/(L+1))

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
# @load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ

lsI2 = zeros(ComplexF64,length(lsρ),length(lsk))

H = TrivialHamiltonian(Latt;params...,returnnode = true)

for (ik,k) in enumerate(lsk)
    @show ik,k
mpo1 = let LocalSpace = TrivialSpinOneHalf,Root = InteractionTreeNode()
    for i in 1:size(Latt)
        addIntr!(Root,(LocalSpace.S₊,),(i,),("S₊",),(false,),1.0 * exp(1im * k * i) / sqrt(L),nothing)
    end
    AutomataSparseMPO(Root,size(Latt))  
end

mpo2 = let LocalSpace = TrivialSpinOneHalf,Root = InteractionTreeNode()
    for i in 1:L
        for pair in neighbor(Latt,i)
            if pair[1] == i
                addIntr!(Root,(LocalSpace.S₋,LocalSpace.Sz),pair,("S₋","Sz"),(false,false),-J*Δ * exp(-1im * k * i) / sqrt(L),nothing)
                addIntr!(Root,(LocalSpace.Sz,LocalSpace.S₋),pair,("Sz","S₋"),(false,false),J * exp(-1im * k * i) / sqrt(L),nothing)
            else
                addIntr!(Root,(LocalSpace.S₋,LocalSpace.Sz),pair,("S₋","Sz"),(false,false),J * exp(-1im * k * i) / sqrt(L),nothing)
                addIntr!(Root,(LocalSpace.Sz,LocalSpace.S₋),pair,("Sz","S₋"),(false,false),-J*Δ * exp(-1im * k * i) / sqrt(L),nothing)
            end
            addIntr!(Root,(LocalSpace.S₋,),(i,),("S₋",),(false,),Hz,nothing)
        end
    end
    AutomataSparseMPO(Root,size(Latt))  
end

for (iρ,ρ) in enumerate(lsρ)
    Env = Environment([mpo1,ρ,mpo2,ρ'])
    initialize!(Env)
    lsI2[iρ,ik] += _scalar(Env) / size(Latt)
end
@show lsI2[end-10,ik]
end


# lsI2