
using TensorKit
dataname = "examples/Heisenberg/data/trivial"

mutable struct InteractionTreeLeave{N}
    A::NTuple{N,AbstractTensorMap}
    site::NTuple{N,Int64}
    name::NTuple{N,String}
    strength::Number
    Z::Union{Nothing,AbstractTensorMap}

    function InteractionTreeLeave(A::NTuple{N,AbstractTensorMap},
            site::NTuple{N,Int64},
            name::NTuple{N,String},
            strength::Number,
            Z::Union{Nothing,AbstractTensorMap} = nothing) where N
        return new{N}(A,site,name,strength,Z)
    end

    function InteractionTreeLeave(A::AbstractTensorMap,
            site::Int64,
            name::String,
            strength::Number,
            Z::Union{Nothing,AbstractTensorMap} = nothing)
        return new{1}((A,),(site,),(name,),strength,Z)
    end
end

include("../../../src/TenetKit.jl")
include("../model.jl")



D = 2^5
Lx = 4
Ly = 1

Latt = YCSqua(Lx,Ly)
J = 1
# for H in [1,2]
H = 1
params = (J=J, H = H)

H =  let Root = InteractionTreeNode(), LocalSpace=TrivialSpinOneHalf
    
    for pair in neighbor(Latt)
        addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋),pair,("S₊","S₋"),J/2,nothing)
        addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊),pair,("S₋","S₊"),J/2,nothing)
        addIntr!(Root,LocalSpace.SzSz,pair,("Sz","Sz"),J,nothing)
    end

    for i in 1:size(Latt)
        addIntr!(Root,LocalSpace.Sz,i,"Sz",-H,nothing)
    end

    InteractionTree(Root)
end
# H.Root.children[1].children[1].children[1].Leave
# H.Root.children[end].children[end].Leave

# for l in Leaves(H.Root)
#     @show l.Leave
# end