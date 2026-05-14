
# using TensorKit
using LinearAlgebra: tr,dot,diagm,eigen
using FiniteLattices, JLD2
# include("../../src/TenetKit.jl")
# include("model.jl")

get_cellsize(Latt::CompositeLattice) = map(x -> maximum([Latt.subLatts[1].sites[ii][x] for ii in 1:div(size(Latt),length(Latt.subLatts))]),1:2)
get_cellsize(Latt::SimpleLattice) = map(x -> maximum([Latt.sites[ii][x] for ii in 1:size(Latt)]),1:2)

dataname = "examples/Heisenberg/data/ed"
function YCRect(L::Int64, W::Int64, (a,b)::NTuple{2,Float64} = (1.0,1.0),θ::Real = 0.0)
    # @assert L ≥ W
    e = ((a, 0.0), (0.0, b))
    sites = [(x, y) for x in 1:L for y in 1:W]
    if iszero(θ)
         BC = PeriodicBoundaryCondition((0, W))
    else
         BC = TwistBoundaryCondition((0, W), θ)
    end
    return SquareLattice(e, sites, BC)
end

Lx = 8
Ly = 1
Latt = YCRect(Lx,Ly)
@save "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt


d = 2
L = size(Latt)

Sx = [0 1;1 0] / 2
Sy = [0 -1im;1im 0] / 2 
Sz = [1 0;0 -1] / 2 
S₊ = Sx + 1im * Sy
S₋ = Sx - 1im * Sy

function addIntr2(O1::Matrix,O2::Matrix,i::Int64,j::Int64,L::Int64,d::Int64)
    ident = diagm(ones(2))
    @assert i <= j 
    H = i == 1 ? O1 : ident
    for s in 2:L
        if s == i 
            H = kron(H,O1)
        elseif s == j
            H = kron(H,O2)
        else
            H = kron(H,ident)
        end
    end
    return H
end

function addIntr1(O1::Matrix,i::Int64,L::Int64,d::Int64)
    ident = diagm(ones(2))
    H = i == 1 ? O1 : ident
    for s in 2:L
        if s == i 
            H = kron(H,O1)
        else
            H = kron(H,ident)
        end
    end
    return H
end

τ = 0.5
Nhot = -20
Nxup = -1
βmax = 10
params = (J = 1.0, Δ = 1.0, Hz = 1.0)

H = let H = zeros(d^size(Latt),d^size(Latt)), J = params.J, Δ = params.Δ, Hz = params.Hz
    for pair in neighbor(Latt)
        H += addIntr2(J * Sx,Sx,pair...,size(Latt),d)
        H += addIntr2(J * Sy,Sy,pair...,size(Latt),d)
        H += addIntr2(Δ * J * Sz,Sz,pair...,size(Latt),d)
    end
    for i in 1:size(Latt)
        H += addIntr1(- Hz * Sz,i,size(Latt),d)
    end
    H
end
S₊s = map(x -> addIntr1(S₊,x,L,d), 1:L)
HS₋s = map(x -> addIntr1(S₋,x,L,d) |> y -> H*y - y*H, 1:L)

lsβ = vcat((1.0 + τ) .^ (Nhot:1:Nxup), 1:τ:βmax)

lsβ2 = lsβ[2:end]*2
@save "$(dataname)/lsβ_$(Lx)x$(Ly)_$(params).jld2" lsβ
@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(params).jld2" lsβ2

lsdata = Dict[]
for β in lsβ2
    Z = tr(exp(- β * H))
    ρ = exp(- β * H) / Z
    ρ′ = exp(- β/2 * H)
    E = tr(ρ * H)
    F =  - log(Z) / β
    I = map(x -> β*tr(ρ′ * HS₋s[x] * ρ′ * S₊s[x]) / Z, 1:L)
    data = Dict(
        "F" => F,
        "E" => E,
        "I" => I
    )
    push!(lsdata,data)
end

@save "$(dataname)/lsdata_$(Lx)x$(Ly)_$(params).jld2" lsdata
