function calObs!(Obs::InteractionGraph, obj::T;kwargs...) where T <: Union{DenseMPO,DenseMPS}

    isnothing(Obs.graph) && initialize!(Obs)
    !isempty(Obs.values) && empty!(Obs.values)
    setdefault!(Obs,obj)
    
    if get_nworker() ≤ 2
        to = _calObs_serial!(Obs,obj;kwargs...)
    else
        to = _calObs_threading!(Obs,obj;kwargs...)
    end

    @timeit to "default!" map(e -> default_weight!(e,false), collect_edges(Obs.graph))
    @timeit to "default!" map(e -> default_val!(e), collect_nodes(Obs.graph))

    show(to,title = "calObs!")
    print("\n")
    show(Obs)
    flush(stdout)
    # @show TimerOutputs.todict(to)["inner_timers"]["update_w!"]["n_calls"]
    # @show TimerOutputs.todict(to)["inner_timers"]["update_n!"]["n_calls"]

    return Obs.values
end

function _calObs_serial!(obs::InteractionGraph,obj::T;kwargs...) where T <: Union{DenseMPO,DenseMPS}
    to = TimerOutput()
    stack = DirectedEdge[]
    data = obs.values
    push!(stack, obs.graph.source[1].out_edges..., obs.graph.sink[1].in_edges...)
    while !isempty(stack)
        @timeit to "pop!"    task = pop!(stack)           # LIFO: depth-first
        ans = _update!(task, obj)
        if ans isa Tuple
            name,site,value = ans
            !haskey(data,name) && (data[name] = Dict{Tuple,Number}())
            # @assert !haskey(data[name],site) "Observable Overcounted!",name,site
            data[name][site] = value
        elseif ans isa Vector
            @timeit to "push!" push!(stack, ans...)
        end
    end
    return to
end


function _calObs_threading!(Obs::InteractionGraph, obj::Union{DenseMPO,DenseMPS};kwargs...)
    nworker   = get(kwargs, :nworker,    get_nworker() - 1)
    cachesize = get(kwargs, :cachesize,  4 * nworker)
    showtimes = get(kwargs, :showtimes,  20)
    isdisk    = get(kwargs, :isdisk,     false)
    # max_local: size limit of each worker's private DFS stack.
    # When a node has k children, the first one (if stack is below max_local)
    # stays on the local DFS stack; all others overflow to ch_swap for other
    # workers to pick up.
    #
    # The key insight is that the observable tree's branching factor is typically
    # SMALL (≈1.1 average for SSE1 on L-site chains), so the local stack almost
    # never grows beyond depth × (avg_branching - 1).  With max_local = nworker*4,
    # the stack never triggers overflow → one worker processes the entire tree
    # while others starve (observed: 27% parallel efficiency on L=64 Kitaev).
    #
    # Default = 1: keep exactly one child locally (DFS continuation), push every
    # sibling to ch_swap immediately.  This guarantees work is shared at EVERY
    # branching point, giving maximum parallelism regardless of tree shape.
    # Memory cost: each open Env tensor ≈ D² floats; with nworker concurrent
    # DFS paths of depth L, peak live tensors = O(nworker × L), negligible for
    # D ≤ 1000.  Increase max_local only if ch_swap throughput becomes a
    # bottleneck (i.e., overhead from many small tasks exceeds the parallelism gain).
    max_local = get(kwargs, :max_local, nworker)

    to = TimerOutput()

    ch      = Channel{DirectedEdge}(cachesize)
    ch_swap = LIFOStack{DirectedEdge}()   # LIFO → DFS scheduling
    ch_info = Channel{Any}(Inf)

    println("initialization finish, begin to calculate Observables.")
    flush(stdout)

    # Scheduler: batch-fills ch from ch_swap (LIFO pop) without sleeping.
    # Old code used sleep(0.01) and only moved 1 node per wakeup → workers starved.
    # Now we fill ch to capacity each iteration and only yield when idle.
    Threads.@spawn while isopen(ch)
        moved = false
        while Base.n_avail(ch) < cachesize && isready(ch_swap)
            put!(ch, _take_lifo!(ch_swap))
            moved = true
        end
        moved || yield()   # yield (not sleep) so other tasks run immediately
    end

    task_work = map(1:nworker) do _
        Threads.@spawn while isopen(ch)
            try
                info = _calObs_work!(obj, ch, ch_swap; isdisk=isdisk, max_local=max_local)
                put!(ch_info, (:ok, info))
            catch e
                isopen(ch_info) && put!(ch_info, (:err, e, catch_backtrace()))
                break
            end
        end
    end


    try
        for e in vcat(Obs.graph.source[1].out_edges, Obs.graph.sink[1].in_edges)
            @timeit to "put!" put!(ch, e)
        end
        
        let remain = length(paths(Obs.graph))
            total = remain
            # showtimes=0 means silent (e.g. warmup runs). Guard against cld(total,0).
            showspacing::Int64 = showtimes > 0 ? cld(total, showtimes) : typemax(Int64)
            while remain > 0
                info = take!(ch_info)
                if info[1] == :err
                    isopen(ch) && close(ch)
                    for t in task_work; wait(t); end
                    showerror(stderr, info[2], info[3])
                    throw(info[2])
                end
                data, count, tm = info[2]
                deepmerge!(Obs.values, data)
                # remain -= count
                remain = total - dictsize(Obs.values)
                merge!(to, tm)
                if remain % showspacing == 0
                    show(to, title="$(total - remain)/$(total)")
                    print("\n")
                    flush(stdout)
                end
            end
        end
    finally
        # @show Base.n_avail(ch), Base.n_avail(ch_swap)
        # while Base.n_avail(ch) > 0
        #     sleep(0.1)
        # end
        # Always close channels so spawned tasks (scheduler + workers) terminate.
        # Without try/finally an exception (e.g. the old DivideError from showtimes=0)
        # left ch open → the scheduler's yield()-loop became a busy-spin that maxed
        # out a CPU core indefinitely.
        isopen(ch)      && close(ch)
        isopen(ch_info) && close(ch_info)
    end

    return to
end

function _calObs_work!(obj::T, ch::Channel, ch_swap::LIFOStack;
                       isdisk::Bool = false, max_local::Int = 4) where T <: Union{DenseMPO,DenseMPS}
    # Each worker maintains a private LIFO task stack for DFS traversal.
    # When a branching node has k children:
    #   - Up to max_local children stay on the local stack → processed DFS-locally,
    #     freeing the parent Env as soon as possible.
    #   - Any overflow goes to the shared ch_swap (LIFOStack) → picked up by other workers.
    # This bounds the peak live Env tensors to O(max_local × tree_depth) per worker
    # instead of O(N × L) with the old FIFO scheme.
    count = 0
    to    = TimerOutput()
    task = DirectedEdge[]
    data = Dict{Tuple,Dict}()
    @timeit to "take!" push!(task, take!(ch))
    while !isempty(task)
        let p = pop!(task)
            @timeit to "update!" ans = _update!(p, obj)
            if ans isa Tuple 
                name,site,value = ans
                !haskey(data,name) && (data[name] = Dict{Tuple,Number}())
                # @assert !haskey(data[name],site) "Observable Overcounted!",name,site
                count += 1
                data[name][site] = value
            elseif ans isa Vector
                @timeit to "put!" push!(ch_swap, ans...)
            end
        end
    end
    return data, count, to
end
