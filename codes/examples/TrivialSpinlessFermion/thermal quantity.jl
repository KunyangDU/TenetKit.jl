using TensorKit
include("../../src/TenetKit.jl")
include("model.jl")


Lx = 8
Ly = 1
Latt = YCSqua(Lx,Ly)
L = size(Latt)
#@load "$(dataname)/Latt_$(Lx)x$(Ly).jld2" Latt

params = (μ = 0,)
D = 2^7

tailname = ""
dataname = "examples/TrivialSpinlessFermion/data"

H = Hamiltonian(Latt;params...)

@load "$(dataname)/lsβ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ
@load "$(dataname)/lsρ_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsρ
@load "$(dataname)/lsF_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsF
@load "$(dataname)/lsE_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsE
lsβ2 = lsβ[2:end] * 2

f = zeros(length(lsβ2))
u = zeros(length(lsβ2))
u2 = zeros(length(lsβ2))
u = lsE 
f = lsF
for (i,ρ) in enumerate(lsρ)
    @show lsβ2[i]
    ρH,_ = mul!(deepcopy(ρ),ρ,H;trunc = truncdim(D))
    u2[i] = tr(ρH) 
end
Ce = @. (u2 - u^2) * lsβ2^2

data = Dict(
    "f" => f,
    "u" => u,
    "Ce" => Ce
)

@save "$(dataname)/lsβ2_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" lsβ2
@save "$(dataname)/data_$(Lx)x$(Ly)_$(D)_$(params)$(tailname).jld2" data

lsF .- fe.(lsβ2,Lx,Ly) * L
lsE .- ue.(lsβ2,Lx,Ly) * L
Ce .- ce.(lsβ2,Lx,Ly) * L

