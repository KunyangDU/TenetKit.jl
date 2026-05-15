
function calObs!(Obs::Observable, obj::Union{DenseMPO,DenseMPS};kwargs...)

    Obs.node.cachedict = CachedDict{UInt64, AbstractEnvironmentTensor}(
        round(Int, CACHE_MEMORY_LIMIT[] * OBS_ENV_CACHE_RATIO[]))

    reset_io_timer!()
    if get_num_threads_julia() ≤ 2
        to = _calObs_serial!(Obs,obj;kwargs...)
    else
        to = _calObs_threading!(Obs,obj;kwargs...)
    end

    @timeit to "tree2dict" Obs.values = Dict(Obs)
    merge_io!(to)
    show(to,title = "Observable")
    print("\n")
    flush(stdout)

    get(kwargs,:destroy,true) && (Obs.node = nothing)

    return Obs.values
end

calObs!(Obs::Observable, Env::Environment; kwargs...) = calObs!(Obs.node,Env.layer[1];kwargs...)

function _calObs_threading!(Obs::Observable, obj::Union{DenseMPO,DenseMPS};kwargs...)
    nworker   = get(kwargs, :nworker,    get_num_threads_julia() - 1)
    cachesize = get(kwargs, :cachesize,  4 * nworker)
    showtimes = get(kwargs, :showtimes,  20)
    max_local = get(kwargs, :max_local, 1)

    to = TimerOutput()

    ch      = Channel{AbstractObservableTreeNode}(cachesize)
    ch_swap = LIFOStack{AbstractObservableTreeNode}()   # LIFO → DFS scheduling
    ch_info = Channel{Tuple{Int64,TimerOutput}}(Inf)

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
            info = _calObs_work!(obj, ch, ch_swap; max_local=max_local)
            put!(ch_info, info)
        end
    end

    try
        @timeit to "put!" put!(ch, Obs.node)

        let remain = 0
            map(x -> (remain += 1), Leaves(Obs.node))
            total = remain
            # showtimes=0 means silent (e.g. warmup runs). Guard against cld(total,0).
            showspacing::Int64 = showtimes > 0 ? cld(total, showtimes) : typemax(Int64)
            while remain > 0
                for task in task_work
                    istaskfailed(task) && fetch(task)
                end
                info = take!(ch_info)
                remain -= info[1]
                merge!(to, info[2])
                if remain % showspacing == 0
                    show(to, title="$(total - remain)/$(total)")
                    print("\n")
                    flush(stdout)
                end
            end
        end
    finally
        # Always close channels so spawned tasks (scheduler + workers) terminate.
        # Without try/finally an exception (e.g. the old DivideError from showtimes=0)
        # left ch open → the scheduler's yield()-loop became a busy-spin that maxed
        # out a CPU core indefinitely.
        isopen(ch)      && close(ch)
        isopen(ch_info) && close(ch_info)
    end

    return to
end


function _calObs_serial!(Obs::Observable,obj::Union{DenseMPO,DenseMPS};kwargs...)
    to = TimerOutput()
    # Use a plain Vector as a LIFO stack (DFS) instead of a Channel (FIFO/BFS).
    # With BFS, all tree nodes at depth k are in the queue simultaneously, keeping
    # O(L) parent-Env tensors alive at once. With DFS (LIFO), we finish one branch
    # before starting its siblings, so each parent-Env is freed as soon as the
    # last of its children is processed — peak live Env tensors drops from O(L)
    # to O(depth × branching_factor).
    stack = AbstractObservableTreeNode[]
    push!(stack, Obs.node)
    while !isempty(stack)
        @timeit to "pop!"    task = pop!(stack)           # LIFO: depth-first
        @timeit to "update!" env = _update_node!(task, obj)
        if isempty(task.children)
            task.Leave.value = real(_scalar(env))
            Tuple!(task.Leave)
        else
            for child in task.children
                child.cachedict = task.cachedict
                task.cachedict[child.id] = env
                @timeit to "push!" push!(stack, child)
            end
        end
    end
    return to
end

function _calObs_work!(obj::Union{DenseMPO,DenseMPS}, ch::Channel, ch_swap::LIFOStack;
                       max_local::Int = 1)
    # Each worker maintains a private LIFO task stack for DFS traversal.
    # When a branching node has k children:
    #   - Up to max_local children stay on the local stack → processed DFS-locally,
    #     freeing the parent Env as soon as possible.
    #   - Any overflow goes to the shared ch_swap (LIFOStack) → picked up by other workers.
    # All children receive the parent env through the shared CachedDict.
    count = 0
    to    = TimerOutput()
    task  = AbstractObservableTreeNode[]
    @timeit to "take!" push!(task, take!(ch))
    while !isempty(task)
        let p = pop!(task)
            @timeit to "update!" env = _update_node!(p, obj)

            if isempty(p.children)
                p.Leave.value = real(_scalar(env))
                Tuple!(p.Leave)
                count += 1
            else
                for child in p.children
                    child.cachedict = p.cachedict
                    p.cachedict[child.id] = env

                    if length(task) < max_local
                        push!(task, child)              # stay local: DFS continues
                    else
                        @timeit to "put!" push!(ch_swap, child)  # overflow to shared LIFO stack
                    end
                end
            end
        end
    end
    return count, to
end

function _update_node!(node::AbstractObservableTreeNode,obj::Union{DenseMPO,DenseMPS})
    env = take!(node.cachedict, node.id)
    site = node.A.site
    if isnothing(env)
        AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj[1],obj[1]']))
        return LeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2]))
    else
        return contract(obj[site],node.A,obj[site]',env)
    end
end

function _update_node!(node::CompositeObservableTreeNode{2},obj::Union{DenseMPO,DenseMPS})
    @assert node.A[1].site == node.A[2].site
    site = node.A[1].site
    env = take!(node.cachedict, node.id)
    if isnothing(env)
        AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj[1],obj[1]']))
        return LeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2]))
    else
        return (isnan(node.A[1].strength) ? 1 : node.A[1].strength) * (isnan(node.A[2].strength) ? 1 : node.A[2].strength) * pushright(node.A[1], obj[site],node.A[2], obj[site]',env)
    end
end

function Base.Dict(Obs::Observable)
    data = Dict{Tuple,Dict}()
    nodedata = Dict(Obs.node)
    for k in keys(nodedata)
        if isnothing(get(data, k[1], nothing))
            data[k[1]] = Dict()
        end
        data[k[1]][k[2]] = nodedata[k]
    end
    return data
end

function Base.Dict(Obs::AbstractObservableTreeNode)
    data = Dict{Tuple,Float64}()
    for l in Leaves(Obs)
        data[(l.Leave.name,l.Leave.site)] = l.Leave.value
    end
    return data
end

