using CairoMakie,JLD2


dataname = "../codes/examples/Heisenberg/trivial/data"
D = 64
Lx = 8
Ly = 4
params = (J = 1, Δ = 1)
k = [pi,pi]
t = 5
Nt = 2
@load "$(dataname)/lsSS_$(Lx)x$(Ly)_$(D)_$(params)_$(k/pi)_$(t)_$(Nt).jld2" lsSS

lsSS

