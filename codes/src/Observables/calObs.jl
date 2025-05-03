function calObs!(Obs::MPSObservable, ψ::Union{DenseMPO,DenseMPS}; destroy::Bool = true)
    Obs.values = calObs(ψ,Obs.forest)
    destroy && (Obs.forest = nothing)
end

function calObs!(Obs::MPSObservable, Env::Environment; destroy::Bool = true)
    Obs.values = calObs(Env.layer[1],Obs.forest)
    destroy && (Obs.forest = nothing)
end

function calObs(ψ::Union{DenseMPO{L},DenseMPS{L}},
    Obsf::ObserableForest) where L
    Roots = Obsf.Roots.children
    ObsDict = Dict{String,Dict}()
    Ntot = sum(map(x -> length(x.children),Roots))
    Ndone = 0
    to = TimerOutput()
    for Root in Roots
        localto = TimerOutput()
        tempDict = Dict{Tuple,Float64}()
        for subRoot in Root.children
            cutparent!(subRoot)
            tempDict[subRoot.Opr.name] = let 
                @timeit localto "construct MPO" mpo = AutomataSparseMPO(InteractionTree(subRoot),L)
                @timeit localto "make environment" Env = Environment([ψ, mpo, adjoint(ψ)])
                @timeit localto "initialize" initialize!(Env)
                @timeit localto "scalarize" isapproxreal(scalar(Env))
            end
        end
        ObsDict[Root.Opr.name] = tempDict

        Ndone += length(Root.children)
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



