using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")

#= 
Fermion complexity
=#
dataname = "examples/NCTO/Kitaev/data/ZZHC"

D = 40
Lx = 2
Ly = 2
Ds = 32

# HB = 0.
θ = 0.5 * pi
# ϕ = 0.0 * pi

lsHB  = [0.07]
# lsϕ = 2pi*(0:0.1:0.9)
ϕ₀ = 0.5 * pi
lsϕ = ϕ₀ |> x -> x .+ 2pi*[-1e-3,1e-3]

for HB in lsHB, ϕ in lsϕ
Hx = round(HB*sin(θ)*cos(ϕ);digits = 8)
Hy = round(HB*sin(θ)*sin(ϕ);digits = 8)
Hz = round(HB*cos(θ);digits = 8)

params1_Kitaev = (J1 = 0.0, K1 = -1.0, Γ1 = 0.0, Γ1′ = 0.0)
params23 = (J2xy = 0.0, J3xy = 0.0, J3z = 0.0)
paramsH = (Hx = Hx, Hy = Hy, Hz = Hz)

Latt = ZZHoneyComb(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params1 = let 
    v = collect(params1_Kitaev)
    v1 = PC2Y*v
    (J1xy = v1[1], J1z = v1[2], Jpm = v1[3], Jzpm = v1[4])
end

params = merge(params1,params23,paramsH)
params_Kitaev = merge(params1_Kitaev,params23,paramsH)
println("$(Lx)x$(Ly), D = $(D), \nparams = $(params), \nparams_Kitaev = $(params_Kitaev)")

# pinh = params.pinh .*vcat(repeat([[0.,1.,0.],],2Ly),repeat([[0.,-1.,0.],],2Ly))
# H = TrivialHamiltonian(Latt; params...,pinh=pinh)

# outer_points = filter(x -> length(neighbor(Latt,x;level = 2)) != 6, 1:size(Latt))
# lop = filter(x -> x < size(Latt)/2,outer_points)
# rop = filter(x -> x > size(Latt)/2,outer_points)
# pinsites = vcat(lop,rop)
# pinh = params.pinh .* vcat(repeat([-[1/2,sqrt(3)/2,0.],],length(lop)),repeat([[1/2,sqrt(3)/2,0.],],length(lop)))

H = TrivialHamiltonian(Latt;params...,
shift = [0,1], direction=[[0,1],[sqrt(3)/2,-1/2],[sqrt(3)/2,1/2]])

ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end
@save "$(dataname)/ρ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_HEAD.jld2" ρ

Obs = Observable()
LocalSpace = TrivialSpinOneHalf

for i in 1:size(Latt)
    addObs!(Obs,LocalSpace.Sx,i,"Sx",false,nothing)
    addObs!(Obs,LocalSpace.Sy,i,"Sy",false,nothing)
    addObs!(Obs,LocalSpace.Sz,i,"Sz",false,nothing)
    addObs!(Obs,LocalSpace.Sz2,i,"Sz2",false,nothing)
end

for i in 1:size(Latt),j in i+1:size(Latt)
    addObs!(Obs,LocalSpace.SxSx,(i,j),("Sx","Sx"),(false,false),nothing)
    addObs!(Obs,LocalSpace.SySy,(i,j),("Sy","Sy"),(false,false),nothing)
    addObs!(Obs,LocalSpace.SzSz,(i,j),("Sz","Sz"),(false,false),nothing)
end 

lsβ = vcat(2. .^ (-15:1:-1),1:10)
lsβ2 = lsβ[2:end]*2
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsβ
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params_Kitaev).jld2" lsβ2

SETTN1!(lsβ[1], H, ρ;trunc = truncdim(Ds))
Z = normalize!(ρ) ^ 2 

info = TDVPinfo(log(Z))
lsinfo = []

alg = TDVPalgo(SingleSite(),CBEalgo(randSVD(1.2),DSA(),1,_getdim(truncdim(D) & truncbelow(1e-8))),truncdim(D) & truncbelow(1e-8),0,Inf,TDVPDefaultLanczos,true,false)
# alg = TDVPalgo(DoubleSite(),NoAlgorithm(),truncdim(D) & truncbelow(1e-16),0,Inf,TDVPDefaultLanczos,true,false)

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
        "obs" => calObs!(Obs,Env.layer[1];destroy = false,showtimes = 4),
        "E" => info.E,
        "F" => - info.lnZ / lsβ[i] / 2,
        "info" => info
    )
    
    push!(lsinfo, data["info"])
    push!(lsF, data["F"])
    push!(lsE, data["E"])
    push!(lsdata, data)

    info.err > alg.tol && break
    @save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_$(i).jld2" data
    @save "$(dataname)/ρ_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_HEAD.jld2" ρ
end
# ρ = Env.layer[1]

# @save "$(dataname)/ρ_$(Lx)x$(Ly)_$(D)_$(params)_$(length(lsβ)).jld2" ρ
@save "$(dataname)/lsdata_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_$(length(lsβ)).jld2" lsdata
@save "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_$(length(lsβ)).jld2" lsF
@save "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params_Kitaev)_$(length(lsβ)).jld2" lsE

lsE

end
