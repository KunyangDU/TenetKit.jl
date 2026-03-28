using CairoMakie,JLD2,LaTeXStrings,FiniteLattices,ColorSchemes,LinearAlgebra
include("../../analysis/analysis.jl")
include("../model.jl")
dataname = "../codes/examples/Heisenberg/trivial/data"

D = 128
Lx = 20
Ly = 4
params = (J=1,Δ=1)

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt
# @load "$(dataname)/gsdata_$(Lx)x$(Ly)_$(D)_$(params).jld2" gsdata


# lsky = [0,]
# lsk = [[kx,ky] for kx in lskx, ky in lsky][:]
lskxc = pi * range(0,2,101)

v₀ = [0,1] * pi
v₁ = [1,1] * pi
lsc = 0:0.02:1
lsk = [(v₀ + c*(v₁-v₀)) for c in lsc]
lskr = vcat(0,cumsum(norm.(diff(lsk))))
lsknode = vcat(0,cumsum(norm.(diff(pi .* lsvs))))


lslabel = [L"(0,\pi)",L"(\pi,\pi)"]

t₀ = 0.0
Nt = 40
τ = 0.5
lst = t₀ .+ τ*(0:Nt)
lsω = range(0,1.3 * pi,length(lsc))

dySS = zeros(length(lsk),length(lsω),3)

tnode = [0.0,10.0,20.0]
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
xticks = (lsknode, lslabel)
)

hm = heatmap!(ax,lskr,lsω,sum([dySS[:,:,i] for i in 1:3]);colorrange = (0,8),colormap = :Reds)
# hm = contourf!(ax,lskr,lsω,sum([dySS[:,:,i] for i in 1:3]))

# lines!(ax,lskxc,pi*sin.(lskxc/2),color = :red)
# lines!(ax,lskxc,pi*abs.(sin.(lskxc))/2,color = :red)

vpath,rpath,rnode = vrange(lsvs .* pi,100)
function ωk(k::Vector;J = 1,S = 1/2,a = 1)
    return 4*J*S*sqrt(1-sum(cos.((k .* a[1,1])))^2/4)
end
lines!(ax, rpath,1.1727*ωk.(vpath),color = :grey, linestyle = :dash)

ylims!(ax, 0, 4)
xlims!(ax, extrema(lskr))

Colorbar(fig[1,2],hm;label = L"A(\mathbf{k},\omega)")

resize_to_layout!(fig)
display(fig)
save("Heisenberg/trivial/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/trivial/figures/dynamics_$(Lx)x$(Ly)_$(D)_$(params).png",fig)
