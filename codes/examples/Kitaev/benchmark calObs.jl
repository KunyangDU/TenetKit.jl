using TensorKit, Printf
include("../../src/TenetKit.jl")
include("model.jl")

# ─── 系统参数（与 finite T.jl 一致） ────────────────────────────────────────
Lx     = 4
Ly     = 4
Latt   = ZZHoneyComb(Lx, Ly)
DS     = 2^4
D_bench = 100          # 测试用 bond dimension
τ      = 1.0
Nhot   = -20

params = (K = 1.0, Ha = 0.0, Hb = 0.0, Hc = 0.1)
Hx, Hy, Hz = params.Ha * [1,-1,0] / sqrt(2) +
              params.Hb * [1,1,-2] / sqrt(6) +
              params.Hc * [1,1,1]  / sqrt(3)

Hroot = TrivialHamiltonian(Latt; root=true, params..., Hx=Hx, Hy=Hy, Hz=Hz)
H     = AutomataSparseMPO(Hroot, size(Latt))

# ─── 构建观测量树（模板，每次 benchmark 前 deepcopy）────────────────────────
println("Building SSE1 observable tree...")
flush(stdout)
Obs_template = SSE1(Hroot)

# ─── 从 β=0 出发，跑几步 TDVP 把 bond dimension 涨到 D_bench ────────────────
ρ = let
    AuxSpaces = repeat([ℂ^1,], size(Latt) + 1)
    ρ = IdDenseMPO(TrivialSpinOneHalf.PhySpace, AuxSpaces)
    canonicalize!(ρ, 1)
    ρ
end

lsβ = vcat((1.0 + τ) .^ (Nhot:1:-1), 1:τ:20.0)  # 足够长，让 D 长到 D_bench

SETTN1!(lsβ[1], H, ρ; trunc = truncdim(DS))
Z    = normalize!(ρ) ^ 2
info = TDVPinfo(log(Z))
alg  = TDVPalgo(
    SingleSite(),
    CBEalgo(randSVD(1.2), DSA(), 1, _getdim(truncdim(D_bench) & truncbelow(1e-8))),
    truncdim(D_bench) & truncbelow(1e-8),
    0, Inf, TDVPDefaultLanczos, true, false
)

@time "initialize environment" begin
    Env = Environment([ρ, H, ρ'])
    initialize!(Env)
end

println("\nGrowing bond dimension toward D=$(D_bench) ...")
flush(stdout)
for i in 2:length(lsβ)
    alg.τ = (lsβ[i] - lsβ[i-1]) / 2
    TDVP!(Env, alg, info)
    D_now = maximum(x -> dim(getAuxSpace(x)[1]), Env.layer[1].ts[2:end])
    println("  β=$(round(lsβ[i],digits=3))  D=$(D_now)")
    flush(stdout)
    D_now >= D_bench && break     # 达到目标 bond dimension 就停
end

ρ_bench = Env.layer[1]
D_actual = maximum(x -> dim(getAuxSpace(x)[1]), ρ_bench.ts[2:end])
println("Actual bond dimension: $(D_actual)")

# ─── 环境信息 ───────────────────────────────────────────────────────────────
println("\n" * "─"^60)
println("System : $(size(Latt)) sites  ($(Lx)×$(Ly) ZZHoneyComb)")
println("D_bench: $(D_bench)")
println("Julia threads : $(get_num_threads_julia())")
println("BLAS  threads : $(BLAS.get_num_threads())")
println("─"^60)
flush(stdout)

# ─── JIT 热身 ────────────────────────────────────────────────────────────────
println("Warming up JIT...")
flush(stdout)
let Obs_warmup = deepcopy(Obs_template)
    calObs!(Obs_warmup, ρ_bench; nworker=1, destroy=true, showtimes=1)
end
GC.gc(); GC.gc()
println("Warmup done.\n")
flush(stdout)

# ─── nworker sweep ──────────────────────────────────────────────────────────
# 每次用 deepcopy(Obs_template) 保证从完全相同的干净树出发
nworker_list = [1, 2, 4, get_num_threads_julia() - 1]
results = Dict{Int,Float64}()

for nw in nworker_list
    GC.gc(); GC.gc()          # 两次 GC，尽量清空残留垃圾
    Obs = deepcopy(Obs_template)

    println("\n" * "═"^60)
    println("  nworker = $nw")
    println("═"^60)
    flush(stdout)

    t = @elapsed calObs!(Obs, ρ_bench; nworker=nw, destroy=false, showtimes=2)

    results[nw] = t
    println("  ▶ wall time = $(round(t; digits=2)) s")
    flush(stdout)
end

# ─── 汇总 ───────────────────────────────────────────────────────────────────
println("\n" * "─"^60)
println("nworker │ wall time │ speedup vs nworker=1")
println("────────┼───────────┼─────────────────────")
t1 = results[1]
for nw in sort(collect(keys(results)))
    t  = results[nw]
    sp = round(t1 / t; digits=2)
    @printf("  %4d  │  %6.1f s  │  ×%.2f\n", nw, t, sp)
end
println("─"^60)
flush(stdout)
