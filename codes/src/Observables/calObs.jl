# function calObs!(Obs::Observable, ψ::Union{DenseMPO,DenseMPS}; destroy::Bool = true)
#     Obs.values = calObs(ψ,Obs.node)
#     destroy && (Obs.node = nothing)
# end

# function calObs!(Obs::Observable, Env::Environment; destroy::Bool = true)
#     Obs.values = calObs(Env.layer[1],Obs.node)
#     destroy && (Obs.node = nothing)
# end

# function calObs(ψ::Union{DenseMPO{L},DenseMPS{L}},
#     Obsf::ObserableForest) where L
#     Roots = Obsf.Roots.children
#     ObsDict = Dict{String,Dict}()
#     Ntot = sum(map(x -> length(x.children),Roots))
#     Ndone = 0
#     to = TimerOutput()
#     for Root in Roots
#         localto = TimerOutput()
#         tempDict = Dict{Tuple,Float64}()
#         for subRoot in Root.children
#             cutparent!(subRoot)
#             tempDict[subRoot.A.name] = let 
#                 @timeit localto "construct MPO" mpo = AutomataSparseMPO(InteractionTree(subRoot),L)
#                 @timeit localto "make environment" Env = Environment([ψ, mpo, adjoint(ψ)])
#                 @timeit localto "initialize" initialize!(Env)
#                 @timeit localto "scalarize" isapproxreal(scalar(Env))
#             end
#         end
#         ObsDict[Root.A.name] = tempDict

#         Ndone += length(Root.children)
#         show(localto;title = "$(Ndone) / $(Ntot)")
#         print("\n")
#         flush(stdout)
#         merge!(to,localto)
#     end
#     show(to;title = "Observables ($(Ntot))")
#     print("\n")
#     flush(stdout)
#     return ObsDict
# end



# function calObs(ψ::Union{DenseMPO,DenseMPS},Obsf::ObserableForest)
#     Roots = Obsf.Roots
#     ObsDict = Dict{String,Dict}()
#     Ntot = sum(map(x -> treesize(x),values(Roots)))
#     Ndone = 0

#     # TODO: multithreading

#     to = TimerOutput()
#     for rootname in keys(Roots)
#         localto = TimerOutput()
#         # @timeit localto "fillenv!" fillenv!(Obsf,ψ)
#         @timeit localto "tree2dict" tempDict = tree2dict(Obsf.Roots[rootname],ψ)
#         ObsDict[rootname] = tempDict

#         Ndone += treesize(Obsf.Roots[rootname])
#         show(localto;title = "$(Ndone) / $(Ntot)")
#         print("\n")
#         flush(stdout)
#         merge!(to,localto)
#     end
#     show(to;title = "Observables ($(Ntot))")
#     print("\n")
#     flush(stdout)
#     return ObsDict
# end

# function fillenv!(root::AbstractObservableTreeNode,obj::Union{DenseMPS{L},DenseMPO{L}}) where L
#     initialize!(root,obj)
#     children = root.children
#     for site in 1:L
#         children = fillenv!(children, obj, site)
#     end
#     return root
# end

# function fillenv!(parents::Vector,obj::Union{DenseMPS{L},DenseMPO{L}},site::Int64) where L
#     childrens = Vector{AbstractObservableTreeNode}()
#     for p in parents
#         p.Env = contract(obj[site],DenseMPOTensor(p.A.A),obj[site]',p.parent.Env)
#         push!(childrens,p.children...)
#     end
#     return childrens
# end

# function initialize!(root::AbstractObservableTreeNode,obj::Union{DenseMPS{L},DenseMPO{L}}) where L
#     @assert treeheight(root) == L

#     root.Env = let 
#         AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj[1],obj[1]']))
#         LeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2]))
#     end

#     return nothing
# end

# function tree2dict(root::AbstractObservableTreeNode,obj::Union{DenseMPS{L},DenseMPO{L}}) where L
#     obs = Dict{Tuple,Float64}()
#     initialize!(root,obj)
#     parents = [root,]
#     children = root.children
#     for site in 1:L
#         nodes,obs′ = node2dict(children, obj, site)
#         map(x -> x.Env = nothing, parents)
#         merge!(obs,obs′)
#         parents = children
#         children = nodes
#     end
#     return obs
# end

# function node2dict(parents::Vector,obj::Union{DenseMPS{L},DenseMPO{L}},site::Int64) where L
#     childrens = Vector{AbstractObservableTreeNode}()
#     obs = Dict{Tuple,Float64}()
#     for p in parents
#         p.Env = contract(obj[site],DenseMPOTensor(p.A.A),obj[site]',p.parent.Env)
#         push!(childrens,p.children...)
#         isempty(p.children) && (obs[p.name] = real(_scalar(p.Env)))
#     end
#     return childrens,obs
# end

# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe LIFO stack for DFS-ordered work scheduling.
#
# Why LIFO instead of the original Channel (FIFO)?
# A FIFO ch_swap causes BFS traversal: all siblings at depth k are scheduled
# before any node at depth k+1. This keeps O(L) distinct parent-Env tensors
# alive simultaneously (one per tree level), totalling O(N·L·D²) memory with
# N workers. LIFO (DFS) processes the deepest available sibling first, so a
# parent Env is freed as soon as its subtree is finished — dramatically
# reducing the peak number of live Env tensors.
# ─────────────────────────────────────────────────────────────────────────────
mutable struct LIFOStack{T}
    data::Vector{T}
    lock::ReentrantLock
    LIFOStack{T}() where T = new{T}(Vector{T}(), ReentrantLock())
end

function Base.push!(s::LIFOStack{T}, x::T) where T
    lock(s.lock) do
        push!(s.data, x)
    end
end

# LIFO pop — caller must handle the case where the stack is empty.
function _take_lifo!(s::LIFOStack)
    lock(s.lock) do
        pop!(s.data)
    end
end

Base.isready(s::LIFOStack) = !isempty(s.data)   # approximate (no lock), for scheduling hints only

function calObs!(Obs::Observable, obj::Union{DenseMPO,DenseMPS};kwargs...)

    get(kwargs,:isdisk, false) ? (global NODE_DATA_PATH = mktempdir(".")) : (global NODE_DATA_PATH = nothing)
    
    if get_num_threads_julia() ≤ 2
        to = _calObs_serial!(Obs,obj;kwargs...)
    else
        to = _calObs_threading!(Obs,obj;kwargs...)
    end
    
    @timeit to "tree2dict" Obs.values = Dict(Obs)
    show(to,title = "Observable")
    print("\n")
    flush(stdout)

    get(kwargs,:destroy,true) && (Obs.node = nothing)

    !isnothing(NODE_DATA_PATH) && rm(NODE_DATA_PATH)

    return Obs.values
end

calObs!(Obs::Observable, Env::Environment; kwargs...) = calObs!(Obs.node,Env.layer[1];kwargs...)


# function _calObs_threading!(Obs::Observable, obj::Union{DenseMPO,DenseMPS};kwargs...)
#     nworker = get(kwargs,:nworker,get_num_threads_julia() - 1)
#     cachesize =  get(kwargs,:cachesize,4*nworker)
#     showtimes =  get(kwargs,:showtimes, 20)

#     to = TimerOutput()

#     ch = Channel{Union{AbstractObservableTreeNode,String}}(cachesize)
#     ch_swap = Channel{AbstractObservableTreeNode}(Inf)
#     ch_info = Channel{Tuple{Int64,TimerOutput}}(Inf)

#     Threads.@spawn while isopen(ch)
#         if Base.n_avail(ch) < div(cachesize,2) && isready(ch_swap)
#             put!(ch,take!(ch_swap))
#         else
#             sleep(0.01)
#         end
#     end

#     task_work = map(1:nworker) do _
#         Threads.@spawn while isopen(ch)
#             info = _calObs_work!(obj,ch,ch_swap)
#             put!(ch_info,info)
#         end
#     end

#     @timeit to "put!" put!(ch,Obs.node)

#     let remain = 0
#         map(x -> (remain += 1),Leaves(Obs.node))
#         total = remain
#         showspacing::Int64 = cld(total, showtimes)
#         while remain > 0
#             for task in task_work
#                istaskfailed(task) && fetch(task)
#             end
#             info = take!(ch_info)
#             remain -= info[1]
#             merge!(to,info[2])
#             if remain % showspacing == 0
#                 show(to,title = "$(total - remain)/$(total)")
#                 print("\n")
#                 flush(stdout)
#             end
#         end
#     end

#     map([ch, ch_swap, ch_info]) do CH
#         @assert !isready(CH)
#         close(CH)
#     end

#     return to
# end

function _calObs_threading!(Obs::Observable, obj::Union{DenseMPO,DenseMPS};kwargs...)
    nworker   = get(kwargs, :nworker,    get_num_threads_julia() - 1)
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
            info = _calObs_work!(obj, ch, ch_swap; isdisk=isdisk, max_local=max_local)
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
        @timeit to "update!" _update_node!(task, obj)
        if isempty(task.children)
            task.Leave.value = real(_scalar(task.Env))
            Tuple!(task.Leave)
            task.Env = nothing
        else
            for node in task.children
                node.Env = task.Env                       # share reference, no copy
                @timeit to "push!" push!(stack, node)
            end
            task.Env = nothing
        end
    end
    return to
end

function _calObs_work!(obj::Union{DenseMPO,DenseMPS}, ch::Channel, ch_swap::LIFOStack;
                       isdisk::Bool = false, max_local::Int = 4)
    # Each worker maintains a private LIFO task stack for DFS traversal.
    # When a branching node has k children:
    #   - Up to max_local children stay on the local stack → processed DFS-locally,
    #     freeing the parent Env as soon as possible.
    #   - Any overflow goes to the shared ch_swap (LIFOStack) → picked up by other workers.
    # This bounds the peak live Env tensors to O(max_local × tree_depth) per worker
    # instead of O(N × L) with the old FIFO scheme.
    count = 0
    to    = TimerOutput()
    task  = AbstractObservableTreeNode[]
    @timeit to "take!" push!(task, take!(ch))
    while !isempty(task)
        let p = pop!(task)
            @timeit to "update!" p = _update_node!(p, obj)

            if isempty(p.children)
                p.Leave.value = real(_scalar(p.Env))
                Tuple!(p.Leave)
                p.Env = nothing
                count += 1
            else
                for node in p.children
                    if isdisk
                        # Each child gets its own unique temp file so they can be
                        # deserialized independently without races.
                        (filepath, io) = mktemp(NODE_DATA_PATH)
                        serialize(io, p.Env)
                        close(io)
                        node.Env = filepath
                    else
                        # Share the reference — no copy needed.
                        # p.Env is freed once _update_node! has been called on every child.
                        node.Env = p.Env
                    end

                    if length(task) < max_local
                        push!(task, node)              # stay local: DFS continues
                    else
                        @timeit to "put!" push!(ch_swap, node)  # overflow to shared LIFO stack
                    end
                end
            end
            p.Env = nothing
        end
    end
    return count, to
end

function _update_node!(node::AbstractObservableTreeNode,obj::Union{DenseMPO,DenseMPS})
    site = node.A.site
    if isnothing(node.Env)
        node.Env = let 
            AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj[1],obj[1]']))
            LeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2]))
        end
    else
        if typeof(node.Env) <: String
            filepath = node.Env
            node.Env = deserialize(filepath)
            rm(filepath)
        end
        node.Env = contract(obj[site],node.A,obj[site]',node.Env)
    end

    return node
end

function _update_node!(node::CompositeObservableTreeNode{2},obj::Union{DenseMPO,DenseMPS})
    @assert node.A[1].site == node.A[2].site
    site = node.A[1].site
    if isnothing(node.Env)
        node.Env = let 
            AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj[1],obj[1]']))
            LeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2]))
        end
    else
        if typeof(node.Env) <: String
            filepath = node.Env
            node.Env = deserialize(filepath)
            rm(filepath)
        end
        # remove strength dependence
        node.Env = (isnan(node.A[1].strength) ? 1 : node.A[1].strength) * (isnan(node.A[2].strength) ? 1 : node.A[2].strength) * pushright(node.A[1], obj[site],node.A[2], obj[site]',node.Env)
    end
    return node
end


# function _update_node!(filepath::String,obj::Union{DenseMPO,DenseMPS})
#     node = deserialize(filepath)
#     rm(filepath)
#     return _update_node!(node, obj)
# end


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

