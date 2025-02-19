abstract type AbstractObservable end

mutable struct MPSObservable <: AbstractObservable
    forest::Union{Nothing, AbstractObservableForest}
    values::Union{Nothing, AbstractDict}
    L::Union{Nothing, Int64}

    function MPSObservable(forest::AbstractObservableForest,
        L::Int64)
        return new(forest, L) 
    end

    function MPSObservable(forest::AbstractObservableForest)
        return new(forest, treeheight(obj.forest.Roots) - 2) 
    end

    MPSObservable() = new(ObserableForest(), nothing)
end

function update!(obj::MPSObservable)
    obj.L = treeheight(obj.forest.Roots) - 2
end

function addObs!(Obs::MPSObservable,     
    Opri::Union{AbstractTensorMap,NTuple{2,AbstractTensorMap}},
    site::Union{Int64,NTuple{2,Int64}},
    name::Union{String,NTuple{2,String}},
    Z::Union{Nothing,AbstractTensorMap};ObsName = nothing
    )
    addObs!(Obs.forest, Opri, site, name, Z;ObsName = ObsName)
    update!(Obs)
end

function addObs!(Obs::MPSObservable,     
    Opri::AbstractTensorMap,
    site::Int64,
    name::String,
    Z::Union{Nothing,AbstractTensorMap};ObsName = nothing
    )
    addObs!(Obs.forest, Opri, site, name, Z;ObsName = ObsName)
    update!(Obs)
end

function addObs!(Obs::MPSObservable,     
    Opri::NTuple{2,AbstractTensorMap},
    site::NTuple{2,Int64},
    name::NTuple{2,String},
    Z::Union{Nothing,AbstractTensorMap};ObsName = nothing
    )
    addObs!(Obs.forest, Opri, site, name, Z;ObsName = ObsName)
    update!(Obs)
end

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

function scalar(Env::Environment{3})
    @assert Env.center[1] == Env.center[2]
    contract(action(proj1(Env, Env.center[1]), Env.layer[1].ts[Env.center[1]]), Env.layer[3].ts[Env.center[1]])
end

function contract(A::MPSTensor{3}, B::MPSTensor{3})
    return _inproduct(A,adjoint(B))
end

function contract(A::MPSTensor{3}, B::AdjointMPSTensor{3})
    return @tensor A.A[1,2,3] * B.A[3,1,2]
end

function contract(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4})
    return @tensor A.A[3,1,2,4] * B.A[2,4,3,1]
end

# c⁺ᵢcⱼ, cᵢc⁺ⱼ, c⁺ᵢcᵢ
function addObs!(Obsf::ObserableForest{N},
    Opri::Tuple, Latt::AbstractLattice, k::Vector, name::Tuple,
    Z::Union{Nothing,AbstractTensorMap};ObsName = nothing
    ) where N
    isnothing(ObsName) && (ObsName = string(name[1]...))
    klabel = tuple(k...)

    ind = findfirst(map(x -> x.Opr.name == ObsName,Obsf.Roots.children))

    if isnothing(ind)
        _addBranch!(Obsf,ObsName)
        _addBranch!(Obsf.Roots.children[end],klabel)
        ind = length(Obsf.Roots.children)
    else
        indt = findfirst(map(x -> x.Opr.name ≈ klabel,Obsf.Roots.children[ind].children))
        if isnothing(indt)
            _addBranch!(Obsf.Roots.children[ind],klabel)
        else
            @error "k point already exist"
        end
    end

    addIntr!(Obsf.Roots.children[ind].children[end],Opri[1:2],Latt,k,name[1:2],1,Z)
    for i in 1:size(Latt)
        addIntr!(Obsf.Roots.children[ind].children[end],Opri[3],i,name[3],1 / size(Latt),nothing)
    end
end


# c⁺ = exp(-1im)* ... 
function addIntr!(Root::InteractionTreeNode,
    Opri::Tuple,
    Latt::AbstractLattice,k::Vector,
    name::Tuple,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    L = size(Latt)
    for i in 1:L, j in i+1:L, ind in 1:2 
        addIntr2!(Root,Opri[ind],(i,j),name[ind],(-1)^(ind-1)*strength*exp((-1)^ind*1im*dot(k, coordinate(Latt,i) .- coordinate(Latt,j))) / L,Z)
    end
end

function Base.:≈(A::Tuple,B::Tuple)
    collect(A) ≈ collect(B) && return true 
    return false
end
