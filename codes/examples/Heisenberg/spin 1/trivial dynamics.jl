using TensorKit
include("../../../src/TenetKit.jl")
include("model.jl")

gsdataname = "examples/Heisenberg/spin 1/data/trivial"
dataname = "examples/Heisenberg/spin 1/data/trivial"

D = 3^4
Lx = 64
Ly = 1
params = (J = 1,)

@load "$(gsdataname)/Latt_$(Lx)x$(Ly).jld2" Latt
@load "$(gsdataname)/lsEg_$(Lx)x$(Ly)_$(D)_$(params).jld2" lsEg
@load "$(gsdataname)/ψ_$(Lx)x$(Ly)_$(D)_$(params).jld2" ψ

H = TrivialHamiltonian(Latt;params...)

v₀ = [1,0] * pi
v₁ = [1,0] * 2pi
lsc = [0.0,]
lsk = [(v₀ + c*(v₁-v₀)) for c in lsc]

t₀ = 5.0
τ = 0.5
Nt = 10

xyznames = ["x","y","z"]
symbs = [:Sx,:Sy,:Sz]
ψsymbs = [:ψx,:ψy,:ψz]
 

for k in lsk
    @show k/pi
    for iψ in 1:3
        Op = let Sop = symbs[iψ]
            Root = InteractionTreeNode()
            for i in 1:size(Latt)
                addIntr!(Root,TrivialSpinOne.eval(Sop),i,string(Sop),false,exp(1im * dot(k,coordinate(Latt,i))),nothing)
            end
            AutomataSparseMPO(Root,size(Latt))  
        end


        if isfile("$(dataname)/ψ′_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(t₀).jld2")
            @load "$(dataname)/ψ′_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(0.0).jld2" ψ′
            ψ₀ = deepcopy(ψ′)
            @load "$(dataname)/ψ′_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(t₀).jld2" ψ′
        else
            @assert t₀ ≈ 0
            ψ′ = mul!(deepcopy(ψ),ψ,Op ,1,Algebraalgo(DoubleSite(),NoAlgorithm(),truncdim(D) & truncbelow(1e-12),3,1e-8))[1]
            @save "$(dataname)/ψ′_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(t₀).jld2" ψ′
            ψ₀ = deepcopy(ψ′)
        end

        info = TDVPinfo()
        alg = TDVPalgo(SingleSite(),
            CBEalgo(dynamicSVD(1.2,-1),DSA(),1,D),
            truncdim(D) & truncbelow(1e-12),1im * τ/2,1,TDVPDefaultLanczos,true,false
        )
        @time "initialize environment" begin 
            Env = Environment([ψ′,H,ψ′'])
            initialize!(Env)
        end
        flush(stdout)

        lsSS = zeros(ComplexF64,Nt+1)
        lsSS[1] = inner(ψ′,ψ₀') * exp(1im * t₀ * lsEg[end])

        for i in 1:Nt
            t = t₀ + τ * i
            println("t = $(t)")
            flush(stdout)
            TDVP!(Env, alg, info)
            # @save "$(dataname)/info_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(t).jld2" info
            lsSS[i+1] = inner(Env.layer[1],ψ₀') * exp(1im * t * lsEg[end])
        end
        @save "$(dataname)/ψ′_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(t₀ + Nt*τ).jld2" ψ′
        @save "$(dataname)/lsSS_$(xyznames[iψ])_$(Lx)x$(Ly)_$(D)_$(params)_$(round.(k/pi,digits = 8))_$(t₀)_$(t₀ + Nt*τ).jld2" lsSS
    end
    GC.gc()
end