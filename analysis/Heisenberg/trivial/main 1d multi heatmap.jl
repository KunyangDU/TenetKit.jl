using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/Heisenberg/trivial/data"

D = 32
Lx = 64
Ly = 1
params = (J=1,Δ=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata
# @load "$(dataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
# E0 = lsEg[end]

# lsky = [0,]
# lsk = [[kx,ky] for kx in lskx, ky in lsky][:]
lskxc = pi * range(0,2,101)

v₀ = [0,0] * pi
v₁ = [2,0] * pi
lsc = 0:0.01:1
lsk = [(v₀ + c*(v₁-v₀)) for c in lsc]
lskx = pi * range(0,2,length(lsk))

lsω = range(0,1.3 * pi,100)

t₀ = 0.0
Nt = 20
τ = 0.5
lst = t₀ .+ τ*(0:Nt)

dySS = zeros(length(lsk),length(lsω),3)

tnode = [0.0,10.0]
lsSStotal = [ComplexF64[] for j in eachindex(lsk),i in 1:3]
for (ik,k) in enumerate(lsk),i in 1:length(tnode)-1
    for (in,n) in enumerate(("x","y","z"))
        @load "$(dataname)/lsSS_x_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(tnode[i])_$(tnode[i+1]).jld2" lsSS
        # @load "$(dataname)/lsSS_$(n)_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(tnode[i+1]).jld2" lsSS
        push!(lsSStotal[ik,in],(i == 1 ? lsSS : lsSS[2:end])...)
    end
end
lst
# lsSStotal[1,1]

for (i,k) in enumerate(lsk)
    map(1:3) do x
        dySS[i,:,x] = [-2/pi*imag.(sum(lsSStotal[i,x] .* sin.(lst * ω) .* window.(lst/maximum(lst)))) for ω in lsω]
    end
end

dySS = dySS / size(Latt)

figsize = (width = 300,height = 200)

fig = Figure()
ax = Axis(fig[1,1];figsize...,
title = "AFMH Squa$(Ly)x$(Lx), D=$(D), ϵ=$(round(8/maximum(lst);digits = 3))",
ylabel = L"\hbar\omega/J",
xticks = ([0,pi,2pi],[L"0",L"\pi",L"2\pi"])
)

hm = heatmap!(ax,lskx,lsω,sum([dySS[:,:,i] for i in 1:3]);colorrange = (0,3))
# contourf!(ax,lskx,lsω,sum([dySS[:,:,i] for i in 1:3]);levels = range(0, 6, length = 10))

lines!(ax,lskxc,pi*sin.(lskxc/2),color = :grey, linestyle = :dash)
lines!(ax,lskxc,pi*abs.(sin.(lskxc))/2,color = :grey, linestyle = :dash)

Colorbar(fig[1,2],hm;label = L"A(\mathbf{k},\omega)")
xlims!(ax, extrema(lskx))
ylims!(ax, 0, 4)
resize_to_layout!(fig)
display(fig)

save("Heisenberg/trivial/figures/dynamics_hm_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/trivial/figures/dynamics_hm_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
