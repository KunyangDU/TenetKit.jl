using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/trivial"
DISK_BASEDIR[] = mktempdir(pwd())

Lx = 4
Ly = 4
Latt = YCSqua(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 64
DS = 2^4
τ = 0.5
Nhot = -20
βmax = 10
params = (J = 1.0, Δ = 1.0, Hz = 1.0)
Hroot = TrivialHamiltonian(Latt;returnnode = true,params...)
H = AutomataSparseMPO(Hroot,size(Latt))
Hx,Hy,Hz = 0.,0.,params.Hz
ObsI = SSE1(Latt,Hroot,TrivialSpinOneHalf.Sud([Hx,Hy,Hz])...)

ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces;isdisk = true)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat((1.0 + τ) .^ (Nhot:1:-1), 1:τ:βmax)

lsβ2 = lsβ[2:end]*2
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

SETTN1!(lsβ[1], H, ρ;trunc = truncdim(DS),isdisk = true)
Z = normalize!(ρ) ^ 2 

info = TDVPinfo(log(Z))
lsinfo = []

alg = TDVPalgo(
    SingleSite(),
    CBEalgo(
        randSVD(1.2),
        DSA(),1,D
    ),
    truncdim(D) & truncbelow(1e-8),0,Inf,
    TDVPDefaultLanczos,true,false,true
)
lsF = Float64[]
lsE = Float64[]
lsdata = Dict[]

@time "initialize environment" begin 
    Env = Environment([ρ,H,ρ'];disk = true)
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
        "I" => calObs!(ObsI,Env.layer[1];destroy = false,showtimes = 4),
        # "obs" => calObs!(Obs,Env.layer[1];destroy = false,showtimes = 4),
        "E" => info.E,
        "F" => - info.lnZ / lsβ[i] / 2,
        # "F′" => - log(tr()) / lsβ[i] / 2,
        "info" => info
    )
    
    push!(lsinfo, data["info"])
    push!(lsF, data["F"])
    push!(lsE, data["E"])
    push!(lsdata, data)

    info.err > alg.tol && break
    @save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
    @save "$(dataname)/ρ_$(Lx)x$(Ly)_$(D)_$(params)_HEAD.jld2" ρ

    @time GC.gc()
end
