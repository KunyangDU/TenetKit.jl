# LRUCache 改写大型张量存储 — 实现计划

## Context

当 bond dimension D 增大（D > 1000），MPS + MPO + Environment 的全部张量驻留在内存中会导致 OOM。实际访问模式是：DMRG/TDVP 扫描时仅 2-3 个 site 的张量被活跃使用，其余 L-2 个 site 的张量闲置占用 RAM。利用 LRUCache 将冷张量驱逐到磁盘，需要时再反序列化加载回内存。

核心策略：创建一个 `CachedVector{T}` 类型作为 `Vector{T}` 的透明替代品，内部用 LRU 管理哪些张量驻留在内存、哪些序列化到磁盘。所有现有算法代码基本无需改动。

## 访问模式回顾

| 模式 | 频率 | 示例 |
|------|------|------|
| `ts[site]` 读 | 最高 | `obj.ts[site]`, `Env.layer[x].ts[site]` |
| `ts[site:site+1]` 读 | 高 | `obj.ts[site:site+1]` |
| `ts[site] = v` 写 | 中 | `Env.layer[1].ts[site] = tl` |
| `ts[site:site+1] = [a,b]` 写 | 中 | DMRG pushright!/pushleft! |
| `map(x -> x.ts[site], ...)` | 中 | Environment/push.jl |
| `ts[end]` | 低 | Environment/operations.jl 初始化 |
| `ts[:]` | 极低 | Algebra/axpby.jl 一处 |
| `for t in ts` 迭代 | 无 | 不存在 |

## 设计决策

### 驱逐策略：总是序列化

无法检测 `getindex` 返回的张量引用是否会被调用者就地修改（如 `normalize!(ts[i])`, `rmul!(ts[i], x)`）。因此不做状态判断，**驱逐时无条件序列化到磁盘**。

相比张量收缩 O(D³~D⁴) 的计算量，一次序列化 I/O 可忽略。收益是算法代码零改动，且无需维护脏标记状态机。

### 不缓存 SparseMPO

`SparseMPOTensor` 只存储 `m::Matrix{Union{Nothing,AbstractLocalOperator}}`，是轻量级的算子引用矩阵，不含大张量。`SparseMPO.ts` 保持 `Vector{SparseMPOTensor}`。

### 不修改 ObsTree 的 Env 存储

ObservableTreeNode.Env 已有 `isdisk=true` 时的 `serialize/deserialize` + `rm` 模式（写一次、读一次、删一次）。对 ObsTree 的 DFS 遍历已经是最优的，不需要额外的 LRU 层。

### CachedVector 的 adjoint/deepcopy

**关键约束**：不能返回普通 `Vector{T}`（会全量加载到内存导致 OOM）。必须保持逐条加载的缓存行为。

**adjoint**：`adjoint(CachedVector{MPSTensor})` → 返回 `CachedVector{AdjointMPSTensor}`
- 逐条迭代 `for i in 1:len`，每次只加载一个 source 张量
- 计算 `AdjointMPSTensor(source[i].A')`，写入 target CachedVector
- target 缓存满时自然驱逐到磁盘，峰值内存 = 2× cache_capacity 个张量
- 因为 `'` 运算符创建新张量，结果已经是独立副本

**deepcopy**：`deepcopy(CachedVector{T})` → 返回 `CachedVector{T}`
- 同样逐条迭代，`result[i] = deepcopy(source[i])`
- 每条只加载一个，峰值内存 = 2× cache_capacity 个张量
- 需要重写 `Base.deepcopy_internal` 使 Julia 递归 deepcopy 能正确分发

**adjoint(DenseMPS) / adjoint(DenseMPO) 简化**：
- 原来 `deepcopy(adjoint(A.ts))` 中的 `deepcopy` 是多余的（adjoint 已创建新张量）
- 改为直接用 `adjoint(A.ts)`，不再套 `deepcopy`

---

## 实现步骤

### Step 1: 新建 `src/TensorWrapper/CachedVector.jl`

核心类型定义：

```julia
mutable struct CachedVector{T}
    cache::LRU{Int, T}           # LRU 缓存，key=site index
    len::Int                     # 逻辑长度
    diskdir::String              # 序列化文件目录 (mktempdir)
    cold::Dict{Int, String}      # 已驱逐的索引 → 文件路径
end
```

**构造函数**：
- `CachedVector{T}(len, capacity)` — 空缓存
- `CachedVector{T}(data::Vector{T}, capacity)` — 从已有 Vector 批量加载

**接口实现**：
- `getindex(cv, i::Int)` — 缓存命中直接返回；冷存储命中则反序列化加载
- `setindex!(cv, val, i::Int)` — 写入缓存 + 清除 cold 记录
- `getindex(cv, r::UnitRange)` — 返回普通 Vector（逐个 getindex）
- `setindex!(cv, vals, r::UnitRange)` — 逐个 setindex!
- `getindex(cv, ::Colon)` — 全量加载返回 Vector
- `setindex!(cv, vals, ::Colon)` — 全量写入
- `length`, `firstindex`, `lastindex`, `iterate`, `show`

**驱逐回调**：当 LRU 满时，`on_evict` 回调将 entry 序列化到 `diskdir/tensor_{idx}.bin`，记录到 `cold`。

### Step 2: 修改 `src/Globals.jl`

添加全局配置：
```julia
const CACHE_CAPACITY = Ref(8)  # 默认每个 CachedVector 的热缓存容量
```

### Step 3: 修改 `src/TensorWrapper/AbstractTensor.jl`

添加 CachedVector 的 adjoint 和 deepcopy 方法：

```julia
# adjoint: 逐条处理，返回新的 CachedVector
function Base.adjoint(cv::CachedVector{MPSTensor})
    result = CachedVector{AdjointMPSTensor}(length(cv))
    for i in 1:length(cv)
        result[i] = AdjointMPSTensor(cv[i].A')
    end
    return result
end
# ... 其余 5 个张量类型类似

# deepcopy: 逐条处理，返回新的 CachedVector
function Base.deepcopy(cv::CachedVector{T}) where T
    result = CachedVector{T}(length(cv))
    for i in 1:length(cv)
        result[i] = deepcopy(cv[i])
    end
    return result
end
```

### Step 4: 修改 `src/MPS/AbstractMPS.jl`

- `DenseMPS{L,T}`: `ts::Vector{MPSTensor}` → `ts::CachedVector{MPSTensor}`
- `AdjointMPS{L,T}`: `ts::Vector{AdjointMPSTensor}` → `ts::CachedVector{AdjointMPSTensor}`
- 所有构造函数内部：输入的 Vector 通过 `CachedVector{T}(ts)` 转换
- `adjoint` 方法简化：去掉冗余的 `deepcopy`

```julia
# 原来：
Base.adjoint(A::DenseMPS{L,T}) where {L,T} = AdjointMPS{L,T}(deepcopy(adjoint(A.ts)), deepcopy(A.center))
# 改为（adjoint 已创建独立张量，无需 deepcopy）：
Base.adjoint(A::DenseMPS{L,T}) where {L,T} = AdjointMPS{L,T}(adjoint(A.ts), deepcopy(A.center))
```

### Step 5: 修改 `src/MPO/AbstractMPO.jl`

- `DenseMPO{L}`: `ts::Vector{DenseMPOTensor}` → `ts::CachedVector{DenseMPOTensor}`
- `AdjointMPO{L}`: `ts::Vector{AdjointMPOTensor}` → `ts::CachedVector{AdjointMPOTensor}`
- `SparseMPO{L}`: **不改动**（轻量级）
- `RefMPO{L}`: `ts` 字段类型不变，pointer 的 ts 已变为 CachedVector，共享引用自然生效
- `adjoint` 方法简化（去掉冗余 deepcopy，同 MPS）

### Step 6: 修改 `src/Environment/AbstractEnvironment.jl`

- `Environment{N,L}`: `envs::Union{Nothing,Array{AbstractEnvironmentTensor}}` → `envs::Union{Nothing,CachedVector{AbstractEnvironmentTensor}}`

### Step 7: 修改 `src/Environment/operations.jl`

- `initialize!`: `env.envs = Vector{...}(undef, L+1)` → `env.envs = CachedVector{...}(L+1, CACHE_CAPACITY[])`
- `setdefault!` 中的 `env.envs[1] = ...` 和 `env.envs[end] = ...` 自动走 CachedVector 的 setindex!
- `_scalar` 中类似初始化逻辑的修复

### Step 8: 无需修改 `src/TensorWrapper/TensorWrapper.jl`

`getindex`/`setindex` 委托方法（`obj.ts[i]`）自动分发到 CachedVector。`normalize!` 调用链：
1. `obj.ts[site]` → 从 LRU 返回缓存的同一对象引用
2. `normalize!(wrapper)` → 就地修改 `wrapper.A`，缓存中的引用已反映变更

驱逐时总是序列化，因此就地修改不会丢失。无需改动 TensorWrapper.jl。

### Step 9: 修改 `src/tools/tools.jl`

- 为 `CachedVector{MPSTensor}` 和 `CachedVector{DenseMPOTensor}` 添加 `_maxdim` 方法

### Step 10: 修改 `src/Algebra/operations.jl`

- `_scalar` 中的 `env.envs = Vector{...}(undef, ...)` → `CachedVector{...}(...)`

### Step 11: 修改 `src/TenetKit.jl`

在 `include("Defaults.jl")` 之后、`include("TensorWrapper/TensorWrapper.jl")` 之前添加：
```julia
include("TensorWrapper/CachedVector.jl")
```

---

## 影响范围汇总

| 文件 | 改动程度 |
|------|----------|
| `src/TensorWrapper/CachedVector.jl` | **新建** (~180行) |
| `src/Globals.jl` | +2行 |
| `src/TensorWrapper/AbstractTensor.jl` | +50行 (adjoint + deepcopy 重载) |
| `src/MPS/AbstractMPS.jl` | 类型标注 + 构造器 + adjoint简化 (~10处) |
| `src/MPO/AbstractMPO.jl` | 类型标注 + 构造器 + adjoint简化 (~8处) |
| `src/Environment/AbstractEnvironment.jl` | 类型标注调整 (~3处) |
| `src/Environment/operations.jl` | ~5处 Vector → CachedVector |
| `src/tools/tools.jl` | +2个 _maxdim 方法 |
| `src/Algebra/operations.jl` | ~2处 分配语句 |
| `src/Algebra/axpby.jl` | ~1处 ts[:] 访问 |
| `src/TenetKit.jl` | +1行 include |
| `src/TensorWrapper/TensorWrapper.jl` | **0 改动** |
| `src/Algorithm/*.jl` (DMRG/TDVP/CBE...) | **0 改动** |

## 验证方案

1. **基本正确性**：用 `CACHE_CAPACITY = typemax(Int)`（全在内存）运行现有 DMRG/TDVP 示例，结果与改动前完全一致
2. **驱逐正确性**：用 `CACHE_CAPACITY = 3`（强制频繁驱逐）运行 L=20 的 Heisenberg 链 DMRG，能量与无缓存版本误差 < 1e-10
3. **内存使用**：用 `CACHE_CAPACITY = 4`、D=1000、L=50 运行，峰值内存显著低于 L × D² × 16 bytes × (3层)
4. **多线程安全**：在多线程模式下运行 calObs! 和 DMRG 的 push! 操作，确认 LRU+锁机制正确
