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
#         p.Env = contract(obj.ts[site],DenseMPOTensor(p.A.A),obj.ts[site]',p.parent.Env)
#         push!(childrens,p.children...)
#     end
#     return childrens
# end

# function initialize!(root::AbstractObservableTreeNode,obj::Union{DenseMPS{L},DenseMPO{L}}) where L
#     @assert treeheight(root) == L

#     root.Env = let 
#         AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj.ts[1],obj.ts[1]']))
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
#         p.Env = contract(obj.ts[site],DenseMPOTensor(p.A.A),obj.ts[site]',p.parent.Env)
#         push!(childrens,p.children...)
#         isempty(p.children) && (obs[p.name] = real(_scalar(p.Env)))
#     end
#     return childrens,obs
# end

function calObs!(Obs::Observable, obj::Union{DenseMPO,DenseMPS};kwargs...)
    
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

    return Obs.values
end

calObs!(Obs::Observable, Env::Environment; kwargs...) = calObs!(Obs.node,Env.layer[1];kwargs...)


function _calObs_threading!(Obs::Observable, obj::Union{DenseMPO,DenseMPS};kwargs...)
    nworker = get(kwargs,:nworker,get_num_threads_julia() - 1)
    cachesize =  get(kwargs,:cachesize,4*nworker)
    showtimes =  get(kwargs,:showtimes, 20)

    to = TimerOutput()

    ch = Channel{AbstractObservableTreeNode}(cachesize)
    ch_swap = Channel{AbstractObservableTreeNode}(Inf)
    ch_info = Channel{Tuple{Int64,TimerOutput}}(Inf)

    Threads.@spawn while isopen(ch)
        if Base.n_avail(ch) < div(cachesize,2) && isready(ch_swap)
            put!(ch,take!(ch_swap))
        else
            sleep(0.01)
        end
    end

    task_work = map(1:cachesize) do _
        Threads.@spawn while isopen(ch)
            info = _calObs_work!(obj,ch,ch_swap)
            put!(ch_info,info)
        end
    end

    @timeit to "put!" put!(ch,Obs.node)

    let remain = 0
        map(x -> (remain += 1),Leaves(Obs.node))
        total = remain
        showspacing::Int64 = cld(total, showtimes)
        while remain > 0
            for task in task_work
               istaskfailed(task) && fetch(task)
            end
            info = take!(ch_info)
            remain -= info[1]
            merge!(to,info[2])
            if remain % showspacing == 0
                show(to,title = "$(total - remain)/$(total)")
                print("\n")
                flush(stdout)
            end
        end
    end

    map([ch, ch_swap, ch_info]) do CH
        @assert !isready(CH)
        close(CH)
    end

    return to
end


function _calObs_serial!(Obs::Observable,obj::Union{DenseMPO,DenseMPS};kwargs...)
    to = TimerOutput()
    ch = Channel{AbstractObservableTreeNode}(Inf)
    put!(ch,Obs.node)
    while isready(ch)
        @timeit to "take!" task = take!(ch)
        @timeit to "update!" _update_node!(task,obj)
        if isempty(task.children)
            task.Leave.value = real(_scalar(task.Env))
            Tuple!(task.Leave)
            task.Env = nothing
        else
            for node in task.children 
                node.Env = task.Env
                @timeit to "put!" put!(ch,node)
            end
        end
        task.Env = nothing
    end
    close(ch)
    return to
end

function _calObs_work!(obj::Union{DenseMPO,DenseMPS},ch::Channel,ch_swap::Channel)
    count = 0
    to = TimerOutput()
    task = AbstractObservableTreeNode[]
    @timeit to "take!" push!(task,take!(ch))
    while !isempty(task)
        let p = pop!(task)
            @timeit to "update!" _update_node!(p,obj)

            if isempty(p.children)
                p.Leave.value = real(_scalar(p.Env))
                Tuple!(p.Leave)
                p.Env = nothing
                # p.value = real(_scalar(p.Env))
                count += 1
            else
                for node in p.children 
                    node.Env = p.Env
                    if length(task) < 1
                        push!(task,node)
                    else
                        @timeit to "put!" put!(ch_swap,node)
                    end
                end
            end
            p.Env = nothing
        end
    end
    return count,to
end

function _update_node!(node::AbstractObservableTreeNode,obj::Union{DenseMPO,DenseMPS})
    site = node.A.site
    if isnothing(node.Env)
        node.Env = let 
            AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj.ts[1],obj.ts[1]']))
            LeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2]))
        end
    else
        node.Env = contract(obj.ts[site],node.A,obj.ts[site]',node.Env)
    end
end

function _update_node!(node::CompositeObservableTreeNode{2},obj::Union{DenseMPO,DenseMPS})
    @assert node.A[1].site == node.A[2].site
    site = node.A[1].site
    if isnothing(node.Env)
        node.Env = let 
            AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj.ts[1],obj.ts[1]']))
            LeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2]))
        end
    else
        # remove strength dependence
        node.Env = (isnan(node.A[1].strength) ? 1 : node.A[1].strength) * (isnan(node.A[2].strength) ? 1 : node.A[2].strength) * pushright(node.A[1], obj.ts[site],node.A[2], obj.ts[site]',node.Env)
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

