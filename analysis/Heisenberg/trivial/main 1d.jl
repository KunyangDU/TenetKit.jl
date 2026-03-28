using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/Heisenberg/trivial/data"

D = 2^6
Lx = 10
Ly = 1
params = (J=1,Δ=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
E0 = lsEg[end]

# lsky = [0,]
# lsk = [[kx,ky] for kx in lskx, ky in lsky][:]
lskxc = pi * range(0,2,101)
lskx = pi * (0:0.1:1)

# lsk = [[kx,0] for kx in lskx]
v₀ = [0,0] * pi
v₁ = [1,0] * pi
lsc = 0:0.2:1
lsk = [(v₀ + c*(v₁-v₀)) for c in lsc]

lsω = range(0,1.3 * pi,100)

t = 5
Nt = 11

lst = range(0,t,Nt)

dySS = zeros(length(lsk),length(lsω),3)

for (i,k) in enumerate(lsk)
    @load "$(dataname)/lsSS_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(t)_$(Nt).jld2" lsSS
    i == length(lsk) && @show lsSS[1]
    map(1:3) do x
        dySS[i,:,x] = [2real.(sum(lsSS[x] .* exp.(1im * lst * ω) .* window.(lst/maximum(lst)))) for ω in lsω]
    end
end

dySS = dySS / size(Latt)

figsize = (width = 300,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "AFMH Squa$(Ly)x$(Lx), D=$(D), ϵ=$(round(8/t;digits = 3))",
ylabel = L"\hbar\omega/J",
)

hm = heatmap!(ax,lskx,lsω,sum([dySS[:,:,i] for i in 1:3]);colorrange = (2.5,15),colormap = :Blues)

lines!(ax,lskxc,pi*sin.(lskxc/2),color = :red)
lines!(ax,lskxc,pi*abs.(sin.(lskxc))/2,color = :red)

Colorbar(fig[1,2],hm)

resize_to_layout!(fig)
display(fig)

# save("Heisenberg/trivial/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
# save("Heisenberg/trivial/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


