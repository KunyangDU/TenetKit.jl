using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/Heisenberg/data/trivial"

D = 2^6
Lx = 10
Ly = 1
params = (J=1,)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
@load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg

lskx = 2pi * range(0,1,11)
lsky = [0,]
lsk = [[kx,ky] for kx in lskx, ky in lsky][:]
lskxc = 2pi * range(0,1,101)
lsω = range(0,1.2 * pi,100)

t = 5
Nt = 10

lst = range(0,t,Nt)

dySS = zeros(length(lsk),length(lsω),3)

for (i,k) in enumerate(lsk)
    @load "$(dataname)/SS/lsSS_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k;digits = 3))_$(t)_$(Nt).jld2" lsSS
    map(1:3) do x
        dySS[i,:,x] = [2real.(sum(lsSS[x] .* exp.(1im * lst * ω))) for ω in lsω]
    end
end

dySS = dySS / size(Latt)

figsize = (width = 400,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...)

hm = heatmap!(ax,lskx,lsω,sum([dySS[:,:,i] for i in 1:3]);colormap = :Blues)

lines!(ax,lskxc,pi*sin.(lskxc/2),color = :black)
lines!(ax,lskxc,pi*abs.(sin.(lskxc))/2,color = :black)

Colorbar(fig[1,2],hm)

resize_to_layout!(fig)
display(fig)

save("Heisenberg/SS/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/SS/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).png",fig)


