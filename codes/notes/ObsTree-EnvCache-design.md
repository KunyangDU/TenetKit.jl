# ObsTree 环境 LRU 优化 — 设计文档

## Context

四点关联 / SSE1 等 observable 计算中，ObsTree DFS 遍历时每个内部节点产生一个 `LeftEnvironmentTensor{2}`（O(D²) 大小）。多线程模式下 ch_swap 中的待处理节点持有引用，导致 O(n_worker × L × D²) 的 env 同时存活在内存中，D > 500 / L > 64 时 OOM。

旧 `isdisk` 方案（无条件 serialize/deserialize）废弃不用。

**策略**：给每个节点分配 `id`，所有 env 统一存入 `CachedDict{UInt64, AbstractEnvironmentTensor}`（key = 子节点 id），子节点消费时用自己 id 取回并删除。`CachedDict` 内部用 LRU 管理，热 env 留内存、冷 env 驱逐到磁盘。

`node.Env` 字段直接删除。每个节点持有 `cachedict::CachedDict` 引用，整棵树指向同一个 `CachedDict` 实例。`_update_node!` 通过 `node.cachedict` 取父 env，contract 后返回。

## 访问模式

```
子节点 B (被 pop):
  env = take!(B.cachedict, B.id)              # 取父 env 并删 key（no key → nothing = 根节点）
  B_env = contract(obj.ts[site], B.A, obj.ts[site]', env)  # 收缩得到 B 自己的 env

  if isleaf(B)
      B.Leave.value = real(scalar(B_env))      # 叶节点：标量化
  else
      for child in B.children
          child.cachedict = B.cachedict         # 继承 cachedict 引用
          B.cachedict[child.id] = B_env         # 分发给子节点
          push!(stack, child)
      end
  end
  # B_env 最后一个引用在子节点消费后消失 → GC 回收
```

## 设计决策

### 1. 节点 + id，去掉 Env 字段

```julia
mutable struct ObservableTreeNode
    id::UInt64
    A::Union{Nothing,AbstractLocalOperator}
    parent::Union{Nothing,ObservableTreeNode}
    children::Vector{ObservableTreeNode}
    Leave::Union{Nothing,ObservableTreeLeave}
    cachedict::Union{Nothing,CachedDict}     # 指向共享 CachedDict
end
```

- **`Env` 字段删除** — env 是 `_update_node!` 的返回值，通过 `node.cachedict` 在父子间传递
- **`cachedict` 字段** — 所有节点指向同一个 `CachedDict` 实例，根节点在 `calObs!` 时赋值，子节点在分发时从父节点继承
- `id` 在构造时通过全局计数器分配

`CompositeObservableTreeNode` 同样去掉 `Env` 字段，加 `cachedict` 字段。

### 2. CachedDict — LRU 缓存 + 磁盘驱逐

与 `CachedVector` 同样的 LRU 模式：

```julia
struct EnvEvictHandler
    cache_ref::Ref{Any}
    cold::Dict{UInt64, String}
    diskdir::String
end
function (h::EnvEvictHandler)(k, v)
    haskey(h.cache_ref[], k) && return
    path = joinpath(h.diskdir, "env_$(k).bin")
    serialize(path, v)
    h.cold[k] = path
end

mutable struct CachedDict{K,V}
    cache::LRU{K, V}
    cold::Dict{K, String}
    diskdir::String
    evict::EnvEvictHandler
    lock::ReentrantLock
end
```

接口：

```julia
# 存入（LRU 满时 finalizer 序列化到磁盘）
function Base.setindex!(d::CachedDict, val, key)
    lock(d.lock) do
        d.cache[key] = val
    end
end

# 取出并删除（线程安全）
function take!(d::CachedDict, key)
    lock(d.lock) do
        if haskey(d.cache, key)
            return pop!(d.cache, key)           # 热命中
        elseif haskey(d.cold, key)
            path = d.cold[key]
            val = deserialize(path)             # 冷命中：反序列化
            rm(path)
            delete!(d.cold, key)
            return val
        else
            return nothing                      # 未存入过（根节点）
        end
    end
end
```

- `setindex!`：存入，LRU 满时 finalizer 自动序列化 → `cold[key] = path`
- `take!`：热命中 pop / 冷命中 deserialize + rm / 未命中 nothing

### 3. _update_node! — 通过 node.cachedict 取父 env，返回新 env

```julia
function _update_node!(node, obj)
    env = take!(node.cachedict, node.id)   # 取父 env（无则 nothing = 根节点）
    site = node.A.site
    if isnothing(env)
        AuxSpaces = reverse(map(x -> getAuxSpace(x)[1], [obj.ts[1], obj.ts[1]']))
        return LeftEnvironmentTensor(isometry(AuxSpaces[1], AuxSpaces[2]))
    else
        return contract(obj.ts[site], node.A, obj.ts[site]', env)
    end
end
```

`CompositeObservableTreeNode{2}` 同理。

### 4. 调用方统一逻辑

```julia
# _calObs_work! / _calObs_serial! 核心循环：
let p = pop!(task)
    env = _update_node!(p, obj)          # 返回 contracted env

    if isempty(p.children)
        p.Leave.value = real(_scalar(env))
        Tuple!(p.Leave)
        count += 1
    else
        for child in p.children
            child.cachedict = p.cachedict  # 继承 cachedict 引用
            p.cachedict[child.id] = env    # 分发给子节点
            if length(task) < max_local
                push!(task, child)
            else
                push!(ch_swap, child)
            end
        end
    end
end
```

### 5. calObs! 入口

```julia
function calObs!(Obs::Observable, obj; kwargs...)
    cache_limit = get(kwargs, :cache_limit,
                      round(Int, CACHE_MEMORY_LIMIT[] * OBS_ENV_CACHE_RATIO[]))

    dict = CachedDict{UInt64, AbstractEnvironmentTensor}(cache_limit)
    Obs.node.cachedict = dict     # 根节点持有 dict，子节点在分发时继承

    # ... 原有串行/多线程分发逻辑 ...

    return Obs.values
end
```

- `cache_limit = typemax(Int)`：LRU 永不满，全在内存
- `cache_limit = 0`：每次 `setindex!` 立即驱逐（等价旧 `isdisk=true`）
- 默认值：自适应

删除全部旧 `isdisk` / `NODE_DATA_PATH` / `String` env 逻辑。

## 实现步骤

### Step 1: 修改 `src/Observables/Node.jl`

- `ObservableTreeNode` 加 `id::UInt64`，**删除 `Env` 字段**
- `CompositeObservableTreeNode` 加 `id::UInt64`，**删除 `Env` 字段**
- 构造函数中分配 id（全局计数器）

### Step 2: 新建 `src/Observables/CachedDict.jl`

约 60 行：`EnvEvictHandler` + `CachedDict` + `setindex!` + `take!`。

### Step 3: 修改 `src/Observables/calObs.jl`

- `_update_node!` 从 CACHED_DICT 取 env → contract → **返回** env
- `_calObs_work!` / `_calObs_serial!`：用返回值分发，统一走 CachedDict
- `calObs!` 入口初始化/清理 CACHED_DICT
- 删除所有旧 isdisk / `.Env` 代码

### Step 4: 修改 `src/Globals.jl`

```julia
const OBS_ENV_CACHE_RATIO = Ref(0.15)
```

### Step 5: 修改 `src/TenetKit.jl`

```julia
include("Observables/CachedDict.jl")
```

## 影响范围

| 文件 | 改动 |
|------|------|
| `src/Observables/CachedDict.jl` | **新建** (~60行) |
| `src/Observables/Node.jl` | 加 `id` 字段, **删 `Env` 字段**, 构造器改 |
| `src/Observables/calObs.jl` | `_update_node!` 改签名, `_calObs_work!` / `_calObs_serial!` 重写, `calObs!` 入口, 删 isdisk |
| `src/Globals.jl` | +1行 |
| `src/TenetKit.jl` | +1行 include |

## 验证方案

1. **正确性**：D=32 L=8 运行 SSE1，结果与改动前一致
2. **驱逐正确性**：`cache_limit=0` vs `cache_limit=typemax(Int)`，结果一致
3. **内存**：`cache_limit=4*D²*8`，D=500 L=64 nworker=4，峰值内存低于旧方案
4. **多线程安全**：nworker=4/8，无竞态
