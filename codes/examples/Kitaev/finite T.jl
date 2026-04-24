using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")
dataname = "examples/Kitaev/data"

Lx = 4
Ly = 4
Latt = ZZHoneyComb(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

D = 50
DS = 2^4
τ = 1.0
Nhot = -20
βmax = 20

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.1)
Hx,Hy,Hz = params.Ha * [1,-1,0] / sqrt(2) + params.Hb * [1,1,-2] / sqrt(6) + params.Hc * [1,1,1] / sqrt(3)
Hroot = TrivialHamiltonian(Latt;root = true,params...,Hx = Hx,Hy = Hy,Hz = Hz)
H = AutomataSparseMPO(Hroot,size(Latt))

Obs = SSE1(Hroot)

ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

lsβ = vcat((1.0 + τ) .^ (Nhot:1:-1), 1:τ:βmax)

lsβ2 = lsβ[2:end]*2
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

SETTN1!(lsβ[1], H, ρ;trunc = truncdim(DS))
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
        "I" => calObs!(Obs,Env.layer[1];destroy = false,showtimes = 4),
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

    @time GC.gc()
end

