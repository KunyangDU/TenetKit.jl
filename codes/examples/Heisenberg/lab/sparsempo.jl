using TensorKit

include("../../../src/TenetKit.jl")

# Heisenberg 模型构建函数
include("model.jl")

# ===== 1D Heisenberg 链 =====
Latt = YCRect(6, 1)
ig = TrivialHamiltonian(Latt; J=1.0, returnnode=true)

# 构建 SparseMPO
mpo = AutomataSparseMPO(ig)
println("SparseMPO 构建完成: L = $(length(mpo.ts))")
println()

for i in 1:length(mpo.ts)
    t = mpo[i]
    println("--- Position $i ---")
    println("  dims: DL=$(nsrc(t.left)), D=$(length(t.A)), DR=$(ndst(t.right))")
    println("  operators: ", join(string.(t.A), ", "))
    println("  left  bond: ", t.left)
    println("  right bond: ", t.right)

    # 展开所有路径 (src → op → dst)
    map_1site = compose(t.left, t.right)
    println("  1-site paths ($(length(map_1site)) total):")
    for (a, j, c) in map_1site
        println("    [$a] → $(t.A[j]) → [$c]")
    end
    println()
end

# ===== D 元组 =====
println("D tuple: ", mpo.D)
