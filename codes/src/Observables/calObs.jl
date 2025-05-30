function calObs!(Obs::Observable, ψ::Union{DenseMPO,DenseMPS}; destroy::Bool = true)
    Obs.values = calObs(ψ,Obs.forest)
    destroy && (Obs.forest = nothing)
end

function calObs!(Obs::Observable, Env::Environment; destroy::Bool = true)
    Obs.values = calObs(Env.layer[1],Obs.forest)
    destroy && (Obs.forest = nothing)
end

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
#             tempDict[subRoot.Opr.name] = let 
#                 @timeit localto "construct MPO" mpo = AutomataSparseMPO(InteractionTree(subRoot),L)
#                 @timeit localto "make environment" Env = Environment([ψ, mpo, adjoint(ψ)])
#                 @timeit localto "initialize" initialize!(Env)
#                 @timeit localto "scalarize" isapproxreal(scalar(Env))
#             end
#         end
#         ObsDict[Root.Opr.name] = tempDict

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



function calObs(ψ::Union{DenseMPO,DenseMPS},Obsf::ObserableForest)
    Roots = Obsf.Roots
    ObsDict = Dict{String,Dict}()
    Ntot = sum(map(x -> treesize(x),values(Roots)))
    Ndone = 0

    # TODO: multithreading

    to = TimerOutput()
    for rootname in keys(Roots)
        localto = TimerOutput()
        # @timeit localto "fillenv!" fillenv!(Obsf,ψ)
        @timeit localto "tree2dict" tempDict = tree2dict(Obsf.Roots[rootname],ψ)
        ObsDict[rootname] = tempDict

        Ndone += treesize(Obsf.Roots[rootname])
        show(localto;title = "$(Ndone) / $(Ntot)")
        print("\n")
        flush(stdout)
        merge!(to,localto)
    end
    show(to;title = "Observables ($(Ntot))")
    print("\n")
    flush(stdout)
    return ObsDict
end

function fillenv!(root::ObservableTreeNode,obj::Union{DenseMPS{L},DenseMPO{L}}) where L
    initialize!(root,obj)
    children = root.children
    for site in 1:L
        children = fillenv!(children, obj, site)
    end
    return root
end

function fillenv!(parents::Vector,obj::Union{DenseMPS{L},DenseMPO{L}},site::Int64) where L
    childrens = Vector{ObservableTreeNode}()
    for p in parents
        p.Env = contract(obj.ts[site],DenseMPOTensor(p.Opr.Opri),obj.ts[site]',p.parent.Env)
        push!(childrens,p.children...)
    end
    return childrens
end

function initialize!(root::ObservableTreeNode,obj::Union{DenseMPS{L},DenseMPO{L}}) where L
    @assert treeheight(root) == L

    root.Env = let 
        AuxSpaces = reverse(map(x -> getAuxSpace(x)[1],[obj.ts[1],obj.ts[1]']))
        LeftEnvironmentTensor(isometry(AuxSpaces[1],AuxSpaces[2]))
    end

    return nothing
end

function tree2dict(root::ObservableTreeNode,obj::Union{DenseMPS{L},DenseMPO{L}}) where L
    obs = Dict{NTuple{2,Int64},Float64}()
    initialize!(root,obj)
    parents = [root,]
    children = root.children
    for site in 1:L
        nodes,obs′ = node2dict(children, obj, site)
        map(x -> x.Env = nothing, parents)
        merge!(obs,obs′)
        parents = children
        children = nodes
    end
    return obs
end

function node2dict(parents::Vector,obj::Union{DenseMPS{L},DenseMPO{L}},site::Int64) where L
    childrens = Vector{ObservableTreeNode}()
    obs = Dict{NTuple{2,Int64},Float64}()
    for p in parents
        p.Env = contract(obj.ts[site],DenseMPOTensor(p.Opr.Opri),obj.ts[site]',p.parent.Env)
        push!(childrens,p.children...)
        isempty(p.children) && (obs[p.name] = _scalar(p.Env))
    end
    return childrens,obs
end

