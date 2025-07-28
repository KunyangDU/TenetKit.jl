using TensorKit

include("../../../src/TenetKit.jl")
include("../model.jl")
dataname = "examples/Heisenberg/data/trivial"

function setdefault!(Env::Environment{4})
    if issparse(Env.layer[1]) && issparse(Env.layer[3])
        ρ1 = Env.layer[2]
        ρ2′ = Env.layer[4]
        Env.envs[1] = SparseLeftEnvironmentTensor(isometry(space(ρ2′.ts[1])[4]', space(ρ1.ts[1])[2]))
        Env.envs[end] = SparseRightEnvironmentTensor(isometry(space(ρ1.ts[end])[3]', space(ρ2′.ts[end])[1]))
    end
end


function pushright(Hup::SparseMPO{4}, ρ::DenseMPO{4}, Hdown::SparseMPO{4}, ρ′::AdjointMPO{4}, EnvL::SparseLeftEnvironmentTensor, i::Int64)
    tmpEnvL = Vector{Any}(nothing,mpo.D[site][2])

end

D = 20
Lx = 4
Ly = 1
params = (J = 1, Δ = 0.5)

Latt = YCSqua(Lx,Ly)
@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt


JE = let LocalSpace = TrivialSpinOneHalf, Root = InteractionTreeNode(), J=params.J, Δ = params.Δ
    je = -1im*J^2/2
    for i in 1:size(Latt)-2
        addIntr!(Root,(LocalSpace.S₊, LocalSpace.Sz, LocalSpace.S₋),(i,i+1,i+2),("S₊","Sz","S₋"),(false,false,false),je,nothing)
        addIntr!(Root,(LocalSpace.S₋, LocalSpace.Sz, LocalSpace.S₊),(i,i+1,i+2),("S₋","Sz","S₊"),(false,false,false),-je,nothing)
        addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₊, LocalSpace.S₋),(i,i+1,i+2),("Sz","S₊","S₋"),(false,false,false),-je*Δ,nothing)
        addIntr!(Root,(LocalSpace.Sz, LocalSpace.S₋, LocalSpace.S₊),(i,i+1,i+2),("Sz","S₋","S₊"),(false,false,false),je*Δ,nothing)
        addIntr!(Root,(LocalSpace.S₊, LocalSpace.S₋, LocalSpace.Sz),(i,i+1,i+2),("S₊","S₋","Sz"),(false,false,false),-je*Δ,nothing)
        addIntr!(Root,(LocalSpace.S₋, LocalSpace.S₊, LocalSpace.Sz),(i,i+1,i+2),("S₋","S₊","Sz"),(false,false,false),je*Δ,nothing)
    end
    AutomataSparseMPO(Root,size(Latt))
end

# root = CompositeObservableTreeNode((JE,deepcopy(JE)))
# buildtree!(root)
# Obs = Observable(root)
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsρ
# lsdata = Vector(undef,length(lsρ))
ρ = lsρ[end]

Env = Environment([JE,ρ,JE,ρ'])
Env.envs = Vector{AbstractEnvironmentTensor}(undef, Env.L + 1)
setdefault!(Env)
canonicalize!(Env,size(Latt))

    # initialize!(Env)

# Env.envs[end]

# calObs!(Obs,ρ;destroy = false, cachesize = 4*(get_num_threads_julia() - 1))
# for i in length(lsρ):-1:1
#     ρ = lsρ[i]
#     @show i
#     calObs!(Obs,ρ;destroy = false, cachesize = 4*(get_num_threads_julia() - 1))
#     data = Obs.values
#     @save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
# end

# @save "$(dataname)/lsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsdata

