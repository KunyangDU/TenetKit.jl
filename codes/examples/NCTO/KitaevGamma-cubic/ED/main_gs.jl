using JLD2,TensorKit,FiniteLattices,KrylovKit
# include("../../../../src/TenetKit.jl")
include("../model.jl")
include("model.jl")

dataname = "examples/NCTO/KitaevGamma-cubic/ED/data"


Lx = 3
Ly = 2

Latt = PCHoneyComb(Lx,Ly)

Sx = [0 1;1 0]/2
Sy = [0 -1im;1im 0]/2
Sz = [1 0;0 -1]/2

L = size(Latt)

xbonds,ybonds,zbonds = getxyzbonds(Latt;shift = [1/2,sqrt(3)/2], direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]])

J = 0.0
K = -1.0
Γ = 0.05
Γ′ = 0.0
params = (J = J,K = K,Γ = Γ,Γ′ = Γ′)

Kx = K
Ky = K 
Kz = K 

θ = 0.5 * pi
ϕ = 0.0 * pi


lsh = 0:0.05:0.2
lsm = zeros(length(lsh))
lsE = zeros(length(lsh))

Ŝ = let
    hcx,hcy,hcz = round.(round(sin(θ)*cos(ϕ);digits = 3) * [1,1,-2]/sqrt(6) + round(sin(θ)*sin(ϕ);digits = 3) * [-1,1,0]/sqrt(2) + round(cos(θ);digits = 3) * [1,1,1]/sqrt(3);digits = 3)
    hcx*Sx + hcy*Sy + hcz*Sz
end


flux_Latt = PCTria(Lx-1,Ly)
edge_shift = [1/2,sqrt(3)/2]
d = 1/sqrt(3)
flux_shift = [1,sqrt(3)/3]
direction=[[-1/2,sqrt(3)/2],[1/2,sqrt(3)/2],[1,0]]
fluxsites,fluxdirections,direction = getPBCflux(Latt,flux_Latt,direction;d = d,edge_shift = edge_shift,flux_shift = flux_shift)



@time "make Hamiltonian0" begin
    H₀ = let H = zeros(2^L,2^L)

    for xb in xbonds

        H += addIntr2(xb,(Sx,Sx),L,Kx + J)
        H += addIntr2(xb,(Sy,Sy),L,J)
        H += addIntr2(xb,(Sz,Sz),L,J)

        H += addIntr2(xb,(Sy,Sz),L,Γ)
        H += addIntr2(xb,(Sz,Sy),L,Γ)

        H += addIntr2(xb,(Sx,Sy),L,Γ′)
        H += addIntr2(xb,(Sy,Sx),L,Γ′)
        H += addIntr2(xb,(Sx,Sz),L,Γ′)
        H += addIntr2(xb,(Sz,Sx),L,Γ′)
    end

    for yb in ybonds
        H += addIntr2(yb,(Sy,Sy),L,Ky + J)
        H += addIntr2(yb,(Sx,Sx),L,J)
        H += addIntr2(yb,(Sz,Sz),L,J)

        H += addIntr2(yb,(Sx,Sz),L,Γ)
        H += addIntr2(yb,(Sz,Sx),L,Γ)

        H += addIntr2(yb,(Sx,Sy),L,Γ′)
        H += addIntr2(yb,(Sy,Sx),L,Γ′)
        H += addIntr2(yb,(Sy,Sz),L,Γ′)
        H += addIntr2(yb,(Sz,Sy),L,Γ′)
    end

    for zb in zbonds
        H += addIntr2(zb,(Sz,Sz),L,Kz + J)
        H += addIntr2(zb,(Sy,Sy),L,J)
        H += addIntr2(zb,(Sx,Sx),L,J)

        H += addIntr2(zb,(Sx,Sy),L,Γ)
        H += addIntr2(zb,(Sy,Sx),L,Γ)

        H += addIntr2(zb,(Sz,Sy),L,Γ′)
        H += addIntr2(zb,(Sy,Sz),L,Γ′)
        H += addIntr2(zb,(Sx,Sz),L,Γ′)
        H += addIntr2(zb,(Sz,Sx),L,Γ′)
    end

    H
end
end

M = let M = zeros(2^L,2^L)
    for i in 1:size(Latt)
        M += addIntr1(i,Ŝ,L,1)
    end
    M
end

W = let W = zeros(2^L,2^L)
    Ops = (Sx,Sy,Sz)
    fluxops = map(y -> map(x -> Ops[x],y),fluxdirections)

    for i in eachindex(fluxsites)
        W += addIntr6(fluxsites[i],fluxops[i],L,1)
    end
    W
end


for (ih,h) in enumerate(lsh)

    Sh = let
        hx = round(h*sin(θ)*cos(ϕ);digits = 3)
        hy = round(h*sin(θ)*sin(ϕ);digits = 3)
        hz = round(h*cos(θ);digits = 3)
        hcx,hcy,hcz = round.(hx * [1,1,-2]/sqrt(6) + hy * [-1,1,0]/sqrt(2) + hz * [1,1,1]/sqrt(3);digits = 3)
        hcx*Sx + hcy*Sy + hcz*Sz
    end

    Hh = let H = zeros(2^L,2^L)
        for i in 1:size(Latt)
            H += addIntr1(i,Sh,L,-1)
        end
        H
    end

    H = H₀ + Hh

    @time E,V = eigsolve(H, 1, :SR)
    E = E[1]
    V = V[1]

    lsm[ih] = real(V' * M * V / size(Latt))
    lsE[ih] = real(E)
    @show lsm[ih]
end

eddata = Dict(
    "h" => lsh,
    "E" => lsE,
    "m" => lsm
)

@save "$(dataname)/eddata_diff_$(Lx)x$(Ly)_$(params)_θ=$(θ/pi)_ϕ=$(ϕ/pi).jld2" eddata
