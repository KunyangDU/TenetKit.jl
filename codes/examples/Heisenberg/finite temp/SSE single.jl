using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

# function TimerOutputs.merge!(A::AbstractTreeNode, B::AbstractTreeNode, value::Function = nodevalue)
#     ind = findfirst(x -> isequal(value(x), value(B)), A.children)
#     if !isnothing(ind)
#         for c in B.children
#             merge!(A.children[ind], c)
#         end
#     else
#         B′ = deepcopy(B)
#         B′.parent = nothing
#         addchild!(A,B′)
#         return A
#     end
# end

# function merge!!(A::AbstractTreeNode, B::AbstractTreeNode, value::Function = nodevalue)
#     map(B.children) do root 
#         merge!(A,root,value)
#     end
#     return A
# end

D = 2^6
# for Ly in 4:2:20,Lx in Ly:2:20 
Lx = 64
Ly = 1

Latt = YCSqua(Lx,Ly)
for Hz in 1.0
params = (J=1.0, Δ = 1.0, Hz = Hz)

H = TrivialHamiltonian(Latt;params...,returnnode = true)

# @load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
# @load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

# for (iβ,β) in enumerate(lsβ)
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
lsI = Vector{Dict}(undef,length(lsρ))
for (i,ρ) in enumerate(lsρ)
    calObs!(Obs,ρ;destroy = false,
    cachesize = 2*(get_num_threads_julia() - 1))
    lsI[i] = Obs.values
    GC.gc()
end

@save "$(dataname)/lsI_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsI
@show lsI
end
