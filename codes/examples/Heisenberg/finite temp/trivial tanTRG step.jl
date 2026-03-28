using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

#= 
Fermion complexity
=#
dataname = "examples/Heisenberg/data/trivial"

D = 2^7
Lx = 10
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
params = (J=1.0 , Δ = 1.0, Hz = 1.0)
Latt = YCSqua(Lx,Ly)

H = TrivialHamiltonian(Latt; params...)
ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

# Obs = Observable()
# LocalSpace = TrivialSpinOneHalf

# for i in 1:size(Latt)
#     addObs!(Obs,LocalSpace.Sx,i,"Sx",false,nothing)
#     addObs!(Obs,LocalSpace.Sy,i,"Sy",false,nothing)
#     addObs!(Obs,LocalSpace.Sz,i,"Sz",false,nothing)
#     addObs!(Obs,LocalSpace.Sz2,i,"Sz2",false,nothing)
# end

# for i in 1:size(Latt),j in i+1:size(Latt)
#     addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),(false,false),nothing)
#     addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),(false,false),nothing)
#     addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),(false,false),nothing)
# end 

lsβ = vcat((1.0 + 0.5) .^ (-15:1:-1), 1:0.5:10)
lsβ2 = lsβ[2:end]*2
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

SETTN1!(lsβ[1], H, ρ;trunc = truncdim(2^6))
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
    data = Dict(
        # "I" => I,
        # "obs" => calObs!(Obs,Env.layer[1];destroy = false,showtimes = 4),
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




