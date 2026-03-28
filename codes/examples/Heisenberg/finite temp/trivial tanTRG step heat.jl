using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/trivial"

for Lx in [10,20,40,100]
D = 2^7
# Lx = 20
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (J=1,Δ=0.5)
Latt = YCSqua(Lx,Ly)

H = TrivialHamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

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

lsβ = vcat(2. .^ (-15:1:-1), 1:40)
lsβ2 = lsβ[2:end]*2
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

SETTN1!(lsβ[1], H, ρ;trunc = truncdim(2^5))
Z = normalize!(ρ) ^ 2 

info = TDVPinfo(log(Z))
lsinfo = []

alg = TDVPalgo(SingleSite(),CBEalgo(randSVD(1.2),DSA(),1,_getdim(truncdim(D) & truncbelow(1e-8))),truncdim(D) & truncbelow(1e-8),0,Inf,TDVPDefaultLanczos,true,false)
lsF = Float64[]
lsE = Float64[]
lsdata = Dict[]

@time "initialize environment" begin 
    Env = Environment([ρ,H,ρ'])
    initialize!(Env)
end
flush(stdout)

for i in 2:length(lsβ)
    τ = (lsβ[i]-lsβ[i-1])/2
    println("t = $(lsβ[i])")
    flush(stdout)
    alg.τ = τ
    
    TDVP!(Env, alg, info)

    EnvI = Environment([JE,Env.layer[1],JE,Env.layer[1]'])
    initialize!(EnvI)

    data = Dict(
        "I" => _scalar(EnvI) / size(Latt) * lsβ2[i-1]^2,
        "E" => info.E,
        "F" => - info.lnZ / lsβ[i] / 2,
        "info" => info
    )
    
    push!(lsinfo, data["info"])
    push!(lsF, data["F"])
    push!(lsE, data["E"])
    push!(lsdata, data)

    info.err > alg.tol && break
    @save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    @save "$(dataname)/ρ_$(Lx)x$(Ly)_$(D)_$(params)_HEAD.jld2" ρ
end

@save "$(dataname)/lsdata_$(Lx)x$(Ly)_$(D)_$(params)_$(length(lsβ)).jld2" lsdata
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params)_$(length(lsβ)).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params)_$(length(lsβ)).jld2" lsE
end



