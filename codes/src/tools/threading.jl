# ====================== 统一多线程 dispatcher ======================
# 固定任务数 = get_nworker()（MKL 协调后分给函数级并行的核数），atomic 计数 + @sync/@spawn
# 动态工作窃取只写这一处。闭包捕获坑（counter/accs 都在本函数作用域）由 dispatcher 一次性兜住。
# 三种累加模式只提供两种：
#   A 不相交写：threaded_foreach —— f(item) 闭包写各自输出槽（每槽只写一次，天然无锁）
#   C 归约    ：threaded_reduce! —— per-thread 私有 acc + 末步确定性归约
# B（锁累加）有意不提供，应迁 A/C。

# 模式 A：不相交写。f(item) 并行执行；需要下标时传 eachindex(work) 作为 work 即可。
function threaded_foreach(f, work; ntasks=get_nworker())
    n = length(work)
    if ntasks <= 1 || n <= 1
        for i in 1:n
            f(work[i])
        end
        return nothing
    end
    counter = Threads.Atomic{Int64}(1)
    Threads.@sync for _ in 1:ntasks
        Threads.@spawn while true
            i = Threads.atomic_add!(counter, 1)
            i > n && break
            f(work[i])
        end
    end
    return nothing
end

# 模式 C：per-thread 累加 + 确定性归约。
#   f(item, acc, w) -> acc：把 item 的贡献累加进私有 acc 并返回；w 是 worker 索引（cache 取 per-worker 状态）
#   accs：每 worker 一个累加器（cache 传持久 buffer；普通归约传 [nothing...] 或标量 zeros）
#   combine!(x, y)：末步确定性归约（跳过未参与累加的 nothing 槽）
function threaded_reduce!(f, work, accs::AbstractVector; combine!, ntasks=length(accs))
    n = length(work)
    if ntasks <= 1
        # 注意：这里不能用 acc 这个名字 —— 类型不稳定时（装箱）会被编译器与下面 @spawn 闭包里的
        # acc 合并成同一个 Core.Box，导致多线程任务往共享 Box 里累加（数据竞争）。改名为 acc0。
        acc0 = accs[1]
        for i in 1:n
            acc0 = f(work[i], acc0, 1)
        end
        return acc0
    end
    counter = Threads.Atomic{Int64}(1)
    Threads.@sync for w in 1:ntasks
        Threads.@spawn begin
            acc = accs[w]
            while true
                i = Threads.atomic_add!(counter, 1)
                i > n && break
                acc = f(work[i], acc, w)
            end
            accs[w] = acc
        end
    end
    x = nothing
    for w in 1:ntasks
        a = accs[w]
        a === nothing && continue
        x = x === nothing ? a : combine!(x, a)
    end
    return x
end
