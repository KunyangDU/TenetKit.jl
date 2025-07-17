# using TensorKit

# SU₂ 对称性的表示，也就是自旋空间

# 为了方便，定义几个自旋量子数 (Sectors)
# j½ = SU2Irrep(1//2)
# j1 = SU2Irrep(1)
# j0 = SU2Irrep(0)

# SU2Space(1/2 => 1)

# Sj½ = SU2Space(j½ => 1)
# 两个自旋 1/2 的粒子融合成一个自旋为 1 的粒子
# (j½ ⊗ j½) → j1

# # 未耦合的输入量子数 (输入的粒子)
# uncoupled = (j½, j½)
# # 耦合后的输出量子数 (总的粒子)
# coupled = j1

# # 构造融合树
# # 对于两个输入的情况，中间步骤是空的 ()
# f = FusionTree(uncoupled, coupled, (true,true), ())

# println("一个基本的 FusionTree:")
# println(f)

# # 你可以查看它的属性
# println("未耦合部分 (输入): ", f.uncoupled)
# println("耦合部分 (输出): ", f.coupled)
# println("内部线 (中间步骤): ", f.innerlines)

# 路径 1: 中间自旋为 j0
# uncoupled = (j½, j½, j½)
# coupled = j½
# intermediate_lines = (j0,) # 中间融合结果

# f1 = FusionTree(uncoupled, coupled, (false,false,false),intermediate_lines)
# println("路径 1 (中间自旋=0):")
# println(f1)

# csp = Sj½ ⊗ Sj½ ⊗ Sj½
# W = isometry(fuse(csp),csp)
# for s in sectors(fuse(csp))
#     @show block(W,s)
# end
# block(W,j0)


# 定义张量的空间
# 一个输出 (codomain) 和三个输入 (domain)
# V_out ← V₁ ⊗ V₂ ⊗ V₃
# codomain = SU2Space(j½ => 1)       # 一个维度为1的 j=1/2 空间
# domain = SU2Space(j½=>3) # 三个维度为1的 j=1/2 空间

# # 创建一个随机张量
# # TensorKit 会自动找出所有可能的融合路径 (FusionTree)
# T = TensorMap(randn, codomain, domain)
# println("创建的 TensorMap T:")
# println(T)

# println("\n张量 T 的内部数据结构:")
# # TensorMap 的 .data 字段是一个字典
# # 它的键 (key) 就是 FusionTree
# for (f, b) in T.data
#     println("-------------------------")
#     println("融合树 (FusionTree): ", f)
#     println("对应的数据块 (Block) 的大小: ", size(b))
# end


# 路径 2: 中间自旋为 j1
# intermediate_lines = (j1,) # 中间融合结果

# f2 = FusionTree(uncoupled, coupled, intermediate_lines, ())
# println("\n路径 2 (中间自旋=1):")
# println(f2)




# d = Rep[U₁×SU₂]((0,1//2) => 1, (-1,0) => 1)

# spr = Rep[U₁×SU₂]((0,0) => 1)

# spl = fuse(d',spr)
# isometry(spl ⊗ d,spr)


d = Rep[ℤ₂×SU₂]((0,0) => 2, (1,1//2) => 1)
spr = Rep[ℤ₂×SU₂]((0,0) => 1)
A = isometry(spr'⊗d,fuse(d,spr)')
permute(A,(3,2),(1,))
B = isometry(fuse(d,spr), d' ⊗ spr)
permute(B,(1,2),(3,))
