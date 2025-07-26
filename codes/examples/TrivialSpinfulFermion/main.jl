using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")
# some problems left (up and down's anticommutation)

Lx = 10
Ly = 1

ψ = let 
    AuxSpace = repeat([ℂ^1,], Lx*Ly)
    randMPS(TrivialSpinfulFermion.PhySpace ,AuxSpace)
end

Latt = YCSqua(Lx,Ly)

params = (t = 1, U = 0, μ = 0)
H = Hamiltonian(Latt;params...)
D = 100

lsE,lsinfo = DMRG1!(ψ,H;trunc = truncdim(D) & truncbelow(1e-12))
showQuantSweep(lsE .- ue(100,Lx,Ly)*size(Latt))

# μ = 0
# U = 0
# t = 1

# let 
#     Root = InteractionTreeNode()
#     LocalSpace = TrivialSpinfulFermion

#     for i in 1:size(Latt)
#         addIntr!(Root,LocalSpace.n,i,"n",false,-μ,nothing)
#         addIntr!(Root,LocalSpace.nd,i,"nd",false,U,nothing)
#     end
    
#     for pair in neighbor(Latt)
#         addIntr!(Root,LocalSpace.F₊⁺F₊,pair,("F₊⁺","F₊"),(false,false),-t,LocalSpace.Z)
#         addIntr!(Root,LocalSpace.F₊F₊⁺,pair,("F₊","F₊⁺"),(false,false),t,LocalSpace.Z)
#         addIntr!(Root,LocalSpace.F₋⁺F₋,pair,("F₋⁺","F₋"),(false,false),-t,LocalSpace.Z)
#         addIntr!(Root,LocalSpace.F₋F₋⁺,pair,("F₋","F₋⁺"),(false,false),t,LocalSpace.Z)
#     end

#     Root
# end

#= @time "calculate observables" begin
    Obs = MPSObservable()
    LocalSpace = TrivialSpinfulFermion

    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.n,i,"n",nothing)
    end

#=     for k in -π:π/4:π
        addObs!(Obs.forest, (LocalSpace.F₊⁺F₊,LocalSpace.F₊F₊⁺,LocalSpace.n₊), Latt, [k,0], (("F₊ₖ⁺","F₊ₖ"),("F₊ₖ","F₊ₖ⁺"),"n₊"),LocalSpace.Z)
        addObs!(Obs.forest, (LocalSpace.F₋⁺F₋,LocalSpace.F₋F₋⁺,LocalSpace.n₋), Latt, [k,0], (("F₋ₖ⁺","F₋ₖ"),("F₋ₖ","F₋ₖ⁺"),"n₋"),LocalSpace.Z)
    end =#

    calObs!(Obs,ψ)
end
@show sum([Obs.values["n"][(i,)] for i in 1:size(Latt)])
Obs.values =#
