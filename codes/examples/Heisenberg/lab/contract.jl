include("../../../src/TenetKit.jl")
get_rss_mb() = parse(Int, read(`ps -o rss= -p $(getpid())`, String)) ÷ 1024

d = 4
D = 100

A1 = DenseMPOTensor(TensorMap(randn,ℂ^d ⊗ ℂ^D,ℂ^D ⊗ ℂ^d))
A2 = DenseMPOTensor(TensorMap(randn,ℂ^d ⊗ ℂ^D,ℂ^D ⊗ ℂ^d))
B1 = DenseMPOTensor(TensorMap(randn,ℂ^d ⊗ ℂ^D,ℂ^D ⊗ ℂ^d))'
B2 = DenseMPOTensor(TensorMap(randn,ℂ^d ⊗ ℂ^D,ℂ^D ⊗ ℂ^d))'
EnvL = DenseLeftEnvironmentTensor(LeftEnvironmentTensor(TensorMap(randn,ℂ^D ⊗ ℂ^D, ℂ^D)))
EnvR = DenseRightEnvironmentTensor(RightEnvironmentTensor(TensorMap(randn,ℂ^D, ℂ^D ⊗ ℂ^D)))
peak = 0

@time for i in 1:10
    global peak = max(peak, get_rss_mb())
    contract(EnvL,A1,A2,B1,B2,EnvR)
    GC.gc()
end
println("Peak RSS during loop: $peak MB")
