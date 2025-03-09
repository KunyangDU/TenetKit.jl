function calObs!(Obs::MPSObservable, ψ::DenseMPS; destroy::Bool = true)
    Obs.values = calObs(ψ,Obs.forest)
    destroy && (Obs.forest = nothing)
end

function calObs!(Obs::MPSObservable, Env::Environment; destroy::Bool = true)
    Obs.values = calObs(Env.layer[1],Obs.forest)
    destroy && (Obs.forest = nothing)
end

function calObs(ψ::DenseMPS{L,T},
    Obsf::ObserableForest) where {L,T}
    Roots = Obsf.Roots.children
    ObsDict = Dict{String,Dict}()
    for Root in Roots
        tempDict = Dict{Tuple,Number}()
        for subRoot in Root.children
            cutparent!(subRoot)
            tempDict[subRoot.Opr.name] = let 
                Env = Environment([ψ, AutomataSparseMPO(InteractionTree(subRoot),L), adjoint(ψ)])
                initialize!(Env)
                real(scalar(Env))
            end
        end
        ObsDict[Root.Opr.name] = tempDict
    end
    return ObsDict
end



