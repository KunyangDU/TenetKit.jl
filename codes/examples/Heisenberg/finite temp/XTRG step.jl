using TensorKit
include("../../../src/TenetKit.jl")
include("../model.jl")

dataname = "examples/Heisenberg/data/XTRG"

Lx = 6
Ly = 1

D = 64
DS = 64
algetol = 1e-16
SETTNtol = 1e-16

Latt = YCSqua(Lx,Ly)
L = size(Latt)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params = (J = 1.0, Δ = 1.0, Hz = 0.0)

Obs = let LocalSpace = TrivialSpinOneHalf, Sops = (LocalSpace.Sx,LocalSpace.Sy,LocalSpace.Sz), Snames = ("Sx","Sy","Sz")
    Obs = Observable()
    for i in 1:size(Latt)
        addObs!(Obs,LocalSpace.Sx,i,"Sx",false,nothing)
        addObs!(Obs,LocalSpace.Sy,i,"Sy",false,nothing)
        addObs!(Obs,LocalSpace.Sz,i,"Sz",false,nothing)
    end
    for i in 1:size(Latt), j in 1:size(Latt)
        i > j && continue
        addObs!(Obs,(LocalSpace.Sx,LocalSpace.Sx),(i,j),("Sx","Sx"),(false,false),nothing)
        addObs!(Obs,(LocalSpace.Sy,LocalSpace.Sy),(i,j),("Sy","Sy"),(false,false),nothing)
        addObs!(Obs,(LocalSpace.Sz,LocalSpace.Sz),(i,j),("Sz","Sz"),(false,false),nothing)
    end

    Obs
end

H = TrivialHamiltonian(Latt; params...)

ρ = let 
    AuxSpaces = repeat([ℂ^1,], size(Latt)+1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ,1)
    ρ
end

N = 25

β0 = 2. ^ (-10)
lsβ = [β0 * 2^i for i in 1:N] * 2
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ

SETTN2!(β0, H, ρ;trunc = truncdim(DS),tol = SETTNtol,multo = algetol)

algo = XTRGalgo(
    DoubleSite(),
    Algebraalgo(
        DoubleSite(),
        CBEalgo(dynamicSVD(1.2,2),DDA(),3,D),
        truncdim(D),
        20,algetol
    ),
    N,H
)
info = XTRGinfo( 2 * log(normalize!(ρ)) )

while info.n ≤ algo.N
    _,_,sweepinfo = XTRG!(ρ,algo,info)
    
    data = Dict(
        "E" => info.E,
        "F" => - info.lnZ / lsβ[info.n],
        "obs" => calObs!(Obs,ρ;destroy = false,showtimes = 4)
    )

    @save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(DS)_$(algetol)_$(SETTNtol)_$(params)_$(info.n).jld2" data
    @save "$(dataname)/ρ_$(Lx)x$(Ly)_$(D)_$(params)_HEAD.jld2" ρ

    info.n += 1
end

