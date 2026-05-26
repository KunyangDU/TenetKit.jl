# SparseMPS — 基于图结构的稀疏 MPS 与 2-site DMRG

## Context

SparseMPO v2.0 已实现从最优 DAG 构建稀疏算符矩阵。对称地，SparseMPS 用同样的图结构表示变分子空间：每条 `entry→exit` 路径 = 一个直积态，边权重 = 该态的振幅。目标是用 2-site DMRG 在这个稀疏表示上做变分优化，试验可行性。

核心机制：2-site 稀疏张量做**块对角 SVD**（复用 `sparse_svd`），SVD 后 u/v 直接成为新路径的左右节点，被截断的路径失去 in/out 边后被 L2R+R2L 扫描自然回收。

## 图结构对应

与 SparseMPO 共享同一张图，但节点含义不同：

| | SparseMPO | SparseMPS |
|---|---|---|
| 节点 | 局域算符 (Sx, Sz, ...) | 物理态基 (↑, ↓) |
| 边 | 算符间的环境收缩 | 路径间的振幅传递 |
| 层 | L 层物理算符 | L 层物理指标 |
| entry→exit 路径 | 哈密顿量中的一项 | 一个直积态 |
| 路径权重 | 耦合常数 | 振幅 |

## 数据结构

文件: `examples/lab/Intr/SparseMPS.jl`

```julia
# SparseMPS: L 层稀疏 MPS，与 SparseMPO 共享图结构
mutable struct SparseMPS{L}
    ts::Vector{SparseMPSTensor}     # 各位置张量 (长度 L)
    D::NTuple{L,NTuple{3,Int64}}   # D[i] = (DL, D, DR)
end

# SparseMPSTensor{DL,DR}: 位置 i 的张量
# DL = 左 bond 节点数，DR = 右 bond 节点数，d = 物理维数 (spin-1/2 → 2)
mutable struct SparseMPSTensor{DL,DR}
    # A[σ]: 对每个物理态 σ∈{1,2}，一个 (DL × DR) 的稠密子矩阵
    A::Vector{Matrix{Float64}}             # 长度 d
    validind::NTuple{D,NTuple{2,Vector{Int64}}}   # validind[σ] = (left_indices, right_indices)
    l2r::Vector{Vector{Int64}}             # l2r[p] = 左指标 p 连接的物理态列表
    r2l::Vector{Vector{Int64}}             # r2l[c] = 右指标 c 连接的物理态列表
end
```

## 初始化

```
build_sparse_mps(mpo::SparseMPO{L}) -> SparseMPS{L}

对每个位置 i:
  - DL, DR = mpo.ts[i] 的 l2r 和 r2l 长度
  - 初始全连通: validind[σ] = (1:DL, 1:DR)
  - A[σ] = rand(DL, DR) 随机初始化
  - l2r[p] = [1,2] (所有物理态), r2l[c] = [1,2]
```

初始化后 sweep 过程中 SVD 自然产生稀疏性。

## 2-site DMRG Sweep — 核心算法

### Step 1: 按 (left_bond, right_bond) 分组建块

位置 i 和 i+1 的稀疏张量缩并为块对角形式。块按 `(a_group, b_group)` 分组，其中 a 是位置 i 的左 bond 指标，b 是位置 i+1 的右 bond 指标。

同一个 `(a_group, b_group)` 内的所有路径的中间 bond m 求和后形成一个稠密块：

```
# 对每组 (a_group, b_group):
#   DL_group = 该组包含的左 bond 指标数
#   DR_group = 该组包含的右 bond 指标数
#   块大小 = (DL_group × d) × (d × DR_group)
#   即 N = DL_group × d² × DR_group

block[(a,σ_L), (σ_R,b)] = Σ_m A_i[σ_L][a,m] * A_{i+1}[σ_R][m,b]
```

其中 a ∈ a_group, b ∈ b_group, σ_L,σ_R ∈ {↑,↓}。

自然分组方式：位置 i 的 `r2l` 和位置 i+1 的 `l2r` 定义了中间 bond m 的连通性。相同 `(a_group, b_group)` 的路径共享同一组中间节点，形成独立的块。

### Step 2: 块对角 SVD（复用 sparse_svd）

对每个块做独立 SVD，然后全局排序 + 截断：

```
blocks = [build_block(i, i+1, group) for group in groups]
results = sparse_svd(blocks; D=D_max, tol=ε)
# results[j] = (λ_j, u_j, v_j, block_idx_j)
```

- 每个块独立 SVD：`X_block = U * S * V'`，X_block 大小 (DL_group×d) × (d×DR_group)
- 全局按 λ 降序排列，截断到 D_max 或 tol
- `u_j` 大小 (DL_group×d) × 1，`v_j` 大小 1 × (d×DR_group) — 每个保留的奇异值对应一个新路径
- `block_idx_j` 标识 `u_j, v_j` 来自哪个原始块，用于继承 in/out bond

### Step 3: 路径创建 — u/v 成为新左右节点

每个保留的 `(λ_j, u_j, v_j, block_idx_j)`:

```
# u_j 是 (DL_group×d) 维向量 → 重塑为新左张量 (位置 i)
# 对每个 σ ∈ {↑,↓}:
#   A_i_new[σ][a_group, j] = u_j[idx_range(a_group, σ)]
#   （即从 u_j 中提取对应 a_group 和 σ 的分量）

# v_j 是 (d×DR_group) 维向量 → 重塑为新右张量 (位置 i+1)
# 对每个 σ ∈ {↑,↓}:
#   A_{i+1}_new[σ][j, b_group] = λ_j * v_j[idx_range(σ, b_group)]
```

- 新路径 j 继承 `block_idx_j` 对应的原始 in-bond (a_group) 和 out-bond (b_group)
- 位置 i 的新 out-bond 和位置 i+1 的新 in-bond 由路径 j 的 index 给出
- 块做完 SVD 后直接被覆盖：原 2-site 的路径 → 新 1+1 site 的路径

### Step 4: 路径湮灭 — 被截断的路径自然消亡

SVD 截断后：
- 未被任何保留的 λ_j 覆盖的原始路径 → 其节点失去 in 或 out 边
- L2R + R2L 环境推进扫描时，遇到 in 或 out 为空的节点 → 跳过并回收
- 不需要额外的递归湮灭逻辑

### Step 5: 更新 validind / l2r / r2l

SVD 后扫描非零元重建稀疏结构：

```
for σ in {↑,↓}:
    nonzeros = findall(x -> abs(x) > ε, A_i_new[σ])
    validind[σ] = (unique(row for (row,col) in nonzeros),
                   unique(col for (row,col) in nonzeros))
    l2r[p] = [σ for σ in 1:d if any(nonzeros 中 row==p)]
    r2l[c] = [σ for σ in 1:d if any(nonzeros 中 col==c)]
```

### Step 6: 环境推进

环境推进利用 l2r/r2l 做稀疏缩并，与 SparseMPO 的环境推进对偶：

```
# 右推进 (L→R sweep):
for σ in l2r[p]:    # 左指标 p 连接的物理态
    for c in validind[σ][2]:   # 该物理态连接的右指标
        EnvR_new[c,c'] += A[σ][p,c] * EnvL[p,p'] * A[σ][p',c']
        # 加上 MPO 算符作用:
        for op in mpo_i.validind 中 σ→σ' 的算符:
            EnvR_new[c,c'] += A[σ][p,c] * EnvL[p,p'] * op_element * A[σ'][p',c']
```

## 关键设计决策

1. **N = DL_group × d × d × DR_group**：块大小由左右 bond 组决定，不是固定 d×d。这是核心点。
2. **u, v 直接是新节点张量**：2-site → 2×1-site，做完 SVD 后原块直接覆盖。`idx` (block_idx) 仅用于查 in/out bond 继承关系。
3. **路径湮灭是隐式的**：被截断的路径自然失去边，L2R+R2L 扫描时顺手回收，无需额外递归。
4. **每个保留的奇异值 = 一个新路径**：D_new 个保留的 λ 对应 D_new 条新路径（中间 bond 状态）。

## 实现步骤

### 文件: `examples/lab/Intr/SparseMPS.jl` (新建)

1. `SparseMPSTensor{DL,DR}` 和 `SparseMPS{L}` 结构定义
2. `build_sparse_mps(mpo::SparseMPO{L})` — 从 MPO 初始化，全连通 + 随机值
3. `group_2site_blocks(t1, t2)` — 按 (left_bond, right_bond) 分组建块
4. `pushright_mps!(env, mps, mpo, pos)` / `pushleft_mps!` — 环境推进（利用 l2r/r2l）
5. `dmrg_sweep!(mps, mpo, env; D_max, tol)` — 单次 sweep
   - L→R: 收缩 2-site 块 → sparse_svd → 更新张量 + 重建 validind/l2r/r2l
   - R→L: 同上反向
6. `dmrg_sparse!(mps, mpo; n_sweeps=10, D_max=100, tol=1e-12)` — 主循环

### 验证: `examples/lab/test_sparse_mps.jl` (新建)

在 L=4,6 Heisenberg 链上:
1. `build_sparse_mpo` → `build_sparse_mps` → `dmrg_sparse!`
2. 比较能量与稠密 DMRG 结果
3. 观察 bond dimension / 路径数在 sweep 中的演化

## 文件变更

| 文件 | 操作 |
|------|------|
| `examples/lab/Intr/SparseMPS.jl` | **新建** — 全部 SparseMPS 代码 |
| `examples/lab/Intr/SparseMPO.jl` | 无需修改 |
| `examples/lab/Intr/IntrNode.jl` | 无需修改 |
| `examples/lab/SparseSVD.jl` | 复用 `sparse_svd` |
| `examples/lab/test_sparse_mps.jl` | **新建** — 验证脚本 |
