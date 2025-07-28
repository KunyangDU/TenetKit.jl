using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

D = 2^6
Lx = 8
Ly = 1
params = (J=1, H = 1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(dataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ
@load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata

S = zeros(size(Latt),size(Latt),3)
for i in 1:size(Latt), j in i+1:size(Latt)
    S[i,j,1] = gsdata["SxSx"][(i,j)]
    S[i,j,2] = gsdata["SySy"][(i,j)]
    S[i,j,3] = gsdata["SzSz"][(i,j)]
end

SS = 2*sum(S) + size(Latt)*3/4

function splice!(obj::DenseMPS,Op::AbstractTensorMap)
    for o in obj.ts
        @tensor tmp[-1,-2;-3] ≔ o.A[-1,1,-3] * Op[-2,1]
        o.A = tmp
    end
    return obj
end


ψs = let LocalSpace = TrivialSpinOneHalf
    map([:Sx,:Sy,:Sz]) do x 
        splice!(deepcopy(ψ),LocalSpace.eval(x))
    end
end


# HSx = let LocalSpace = TrivialSpinOneHalf

#     Root = InteractionTreeNode()

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.Sx,i,"Sx",1,nothing)
#     end
#     AutomataSparseMPO(InteractionTree(Root),size(Latt))  
# end

# HSy = let LocalSpace = TrivialSpinOneHalf

#     Root = InteractionTreeNode()

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.Sy,i,"Sy",1,nothing)
#     end
#     AutomataSparseMPO(InteractionTree(Root),size(Latt))  
# end

# HSz = let LocalSpace = TrivialSpinOneHalf

#     Root = InteractionTreeNode()

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.Sz,i,"Sz",1,nothing)
#     end
#     AutomataSparseMPO(InteractionTree(Root),size(Latt))  
# end

# ψSx = mul!(deepcopy(ψ),ψ,HSx,1, truncdim(D) & truncbelow(-Inf) )[1]
# ψSy = mul!(deepcopy(ψ),ψ,HSy,1, truncdim(D) & truncbelow(-Inf) )[1]
# ψSz = mul!(deepcopy(ψ),ψ,HSz,1, truncdim(D) & truncbelow(-Inf) )[1]


# sum(map(ψs) do x
#     inner(x,x')
# end),SS

inner(ψs[2],ψs[2]')
# , inner(ψSz,ψSz')


