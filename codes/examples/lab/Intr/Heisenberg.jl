include("../../../src/TenetKit.jl")
# Lx = 8, Ly = 4 正方晶格上海森堡模型（最近邻S \dot S相互作用）

include("BondMap.jl")
include("IntrNode.jl")
include("IntrGraph.jl")
include("GraphIterator.jl")
include("addIntr.jl")
include("SparseMPO.jl")

Lx = 8
Ly = 4

Latt = YCSqua(Lx,Ly)

for (i,j) in neighbor(Latt)
    addIntr!
end




