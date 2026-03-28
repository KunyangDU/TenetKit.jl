using CairoMakie,JLD2,LsqFit

dataname = "../codes/examples/Heisenberg/data/trivial/scaling"

D = 2^6

Lx = 64
Ly = 1

params = (J=1.0 , Δ = 1.0, Hz = 2.0)
# τ = 0.2
# Nhot = -15
# Tmax = 20

Nmax = 116

Nfit = 70

@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

# params = (J=1.0, Δ = 1.0, Hz = 1.0)
# lsβ = vcat(2. .^ (-15:1:-1), 1:10)
# lsβ = vcat((1.0 + τ) .^ (Nhot:1:-1), 1:τ:Tmax)
# lsβ2 = lsβ[2:end]*2
@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ
@load "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsβ2

# Is = zeros(Nmax-1)
# for i in 2:Nmax
#     @load "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)_$(i).jld2" data
#     Is[i-1] = sum(map(x -> sum(values(x)),values(data["I"]))) * lsβ2[i-1] / size(Latt)
# end

@load "$(dataname)/Is_$(Lx)x$(Ly)_$(D)_$(params).jld2" Is

model(x,p) = @. p[1]*x^p[2]

# fit_curve
fit = curve_fit(model, 1 ./ lsβ2[end-Nfit:end], abs.(Is)[end-Nfit:end],randn(2))
xc = 10. .^ (range(-2,log10(1 / lsβ2[end-Nfit]),100))
yc = model(xc,fit.param)



figsize = (width = 200, height = 200)

fig = Figure()
ax = Axis(fig[1,1];
title = "$(Ly)x$(Lx) Heisenberg, D=$(D)",
# xlabel = L"h_{[111]}",
ylabel = L"\tilde{I}_2",
xlabel = L"T/J",
figsize...,
xscale = log10,
yscale = log10
)

scatterlines!(ax,1 ./ lsβ2[1:Nmax-1],abs.(Is))

lines!(ax,xc,yc;
# linestyle = :dash, 
color = :grey,linewidth = 2)

xlims!(ax,10.0 ^ (-2), 10.0 ^ (1))
# xlims!(ax,1e-8,0.5)

resize_to_layout!(fig)
display(fig)

save("Heisenberg/trivial/figures/SSE_scaling_$(Lx)x$(Ly)_$(D)_$(params).pdf",fig)
save("Heisenberg/trivial/figures/SSE_scaling_$(Lx)x$(Ly)_$(D)_$(params).png",fig)

fit.param