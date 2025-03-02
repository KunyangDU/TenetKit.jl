


mutable struct SparseMPO{L} <: AbstractMPO
    ts::Vector{SparseMPOTensor}
    D::Vector{NTuple{2,Int64}}
    
    function SparseMPO(ts::Vector{SparseMPOTensor},
        D::Vector{NTuple{2,Int64}})
        return new{length(ts)}(ts,D)
    end

    function SparseMPO(ts::Vector{SparseMPOTensor})
        D = map(size,ts)
        return new{length(ts)}(ts,convert(Vector{NTuple{2,Int64}},D))
    end

    function SparseMPO(t::SparseMPOTensor{N,M}) where {N,M}
        D = convert(Vector{NTuple{2,Int64}},[(N,M)])
        ts = convert(Vector{SparseMPOTensor},[t])
        return new{length(ts)}(ts,D)        
    end
end

function Base.length(::SparseMPO{L}) where L 
    return L
end

function issparse(::SparseMPO)
    return true
end


mutable struct DenseMPO{L} <: AbstractMPO
    ts::Vector{DenseMPOTensor}
    center::Vector{Int64}
    
    function DenseMPO(A::Vector{DenseMPOTensor},center::Vector{Int64})
        return new{length(A)}(A,center)
    end

    function DenseMPO(A::Vector{DenseMPOTensor{R}}) where R
        return new{length(A)}(A,[1,length(A)])
    end

    function DenseMPO(t::DenseMPOTensor)
        A = convert(Vector{DenseMPOTensor},[t])
        return new{1}(A,[1,1])        
    end

    function DenseMPO(t::Vector)
        tmp = map(DenseMPOTensor,t)
        A = convert(Vector{DenseMPOTensor},tmp)
        return new{length(A)}(A,[1,length(A)])        
    end
end
const DenseMPQ = Union{DenseMPO,DenseMPS}


function Base.size(t::DenseMPOTensor{4})
    return map(dim,t.A |> x -> (codomain(x)[2],domain(x)[1]))
end

function Base.length(::DenseMPO{L}) where L
    return L
end


mutable struct AdjointMPO{L} <: AbstractMPO
    ts::Vector{AdjointMPOTensor}
    center::Vector{Int64}
    
    function AdjointMPO(A::Vector{AdjointMPOTensor},center::Vector{Int64})
        return new{length(A)}(A,center)
    end

    function AdjointMPO(A::Vector{AdjointMPOTensor{R}}) where R
        return new{length(A)}(A,[1,length(A)])
    end

    function AdjointMPO(t::AdjointMPOTensor)
        A = convert(Vector{AdjointMPOTensor},[t])
        return new{1}(A,[1,1])        
    end

    function AdjointMPO(t::Vector{AbstractTensorMap})
        tmp = map(AdjointMPOTensor,t)
        A = convert(Vector{AdjointMPOTensor},tmp)
        return new{length(A)}(A,[1,length(A)])        
    end
end

function Base.adjoint(A::DenseMPO{L}) where {L}
    return AdjointMPO(deepcopy(adjoint(A.ts)), deepcopy(A.center))
end

function Base.adjoint(A::AdjointMPO{L}) where {L}
    return DenseMPO(deepcopy(adjoint(A.ts)), deepcopy(A.center))
end

issparse(::Union{DenseMPO,AdjointMPO}) = false

function AutomataSparseMPO(Root::InteractionTreeNode,L::Int64=treeheight(Root) - 1)
    MPO = let 
        tempMPO = Vector{SparseMPOTensor}(undef,L)

        idtensor = getIdTensor(Root.children[1].children[1].Opr)

        lastnode = Dict(
            "leaves" => [],
            "roots" => [],
            "inverse_root" => 0,
        )
        nextnode = deepcopy(lastnode)
        lastnode["roots"] = Root.children
        
        for iL in 1:L

            nextnode["leaves"] = []
            nextnode["roots"] = []
            nextnode["leaves_inds"] = []
            nextnode["roots_inds"] = []

            if nextnode["inverse_root"] == 0 && !isempty(findall(x -> isempty(x.children),vcat([lastroot.children for lastroot in lastnode["roots"]]...)))
                nextnode["inverse_root"] = 1
            end

            for (lastind,last_root) in enumerate(lastnode["roots"])
                for next_subtree in last_root.children
                    if isempty(next_subtree.children)
                        push!(nextnode["leaves"],next_subtree)
                        push!(nextnode["leaves_inds"],((lastind + lastnode["inverse_root"], nextnode["inverse_root"]), length(nextnode["leaves"])))
                    else
                        push!(nextnode["roots"],next_subtree)
                        push!(nextnode["roots_inds"],((lastind + lastnode["inverse_root"], length(nextnode["roots"]) + nextnode["inverse_root"]), length(nextnode["roots"])))
                    end
                end
            end

            localMPOdims = length.((lastnode["roots"], nextnode["roots"])) .+ (lastnode["inverse_root"], nextnode["inverse_root"])
            localMPO = SparseMPOTensor(nothing,localMPOdims...)
            localMPO.m[1,1] = DenseMPOTensor(lastnode["inverse_root"]*idtensor)

            map([("leaves_inds","leaves"),("roots_inds","roots")]) do (x,y)
                for inds in nextnode[x]
                    localMPO.m[inds[1]...] += DenseMPOTensor(let 
                        localOpr = nextnode[y][inds[2]].Opr.Opri
                        strength = nextnode[y][inds[2]].Opr.strength
                        if isnan(strength)
                            localOpr
                        else
                            localOpr*strength
                        end
                    end)
                end
            end
            
            lastnode["leaves"] = nextnode["leaves"]
            lastnode["roots"] = nextnode["roots"]
            lastnode["inverse_root"] = nextnode["inverse_root"]

            tempMPO[iL] = localMPO
        end

        SparseMPO(tempMPO)
    end

    return MPO
end

function AutomataSparseMPO(Tree::InteractionTree,L::Int64 = treeheight(Tree.Root) - 1)
    return AutomataSparseMPO(Tree.Root,L)
end

function Base.:+(::Nothing,A::DenseMPOTensor)
    return A
end

function Base.:+(A::DenseMPOTensor,::Nothing)
    return A
end

function _funcDenseMPO(func::Function, PhySpaces::AbstractVector, AuxSpaces::AbstractVector)
    length(PhySpaces) == length(AuxSpaces) && push!(AuxSpaces, trivial(PhySpaces[1]))
    @assert length(PhySpaces) + 1 == length(AuxSpaces)
    tmp = [DenseMPOTensor(TensorMap(func,PhySpaces[i] ⊗ AuxSpaces[i], AuxSpaces[i+1] ⊗ PhySpaces[i])) for i in eachindex(PhySpaces)]
    return DenseMPO(tmp)
end

function _funcDenseMPO(func::Function, PhySpace::ElementarySpace, AuxSpaces::AbstractVector)
    return _funcDenseMPO(func, repeat([PhySpace,],length(AuxSpaces)), AuxSpaces)
end

function IdDenseMPO(PhySpace::ElementarySpace, AuxSpaces::AbstractVector)
    tmp = [DenseMPOTensor(isometry(PhySpace ⊗ AuxSpaces[i], AuxSpaces[i+1] ⊗ PhySpace)) for i in eachindex(AuxSpaces)[1:end-1]]
    return DenseMPO(tmp)
end

function IdDenseMPO(L::Int64, PhySpace::ElementarySpace = ℂ^1, AuxSpace::ElementarySpace = (ℂ^1)')
    return _funcDenseMPO(ones, map(x -> repeat([x,],L),(PhySpace,AuxSpace))...)
end

function RandDenseMPO(L::Int64, PhySpace::ElementarySpace = ℂ^1, AuxSpace::ElementarySpace = (ℂ^1)')
    return _funcDenseMPO(randn, map(x -> repeat([x,],L),(PhySpace,AuxSpace))...)
end


function Base.adjoint(t::CompositeMPOTensor)
    return AdjointCompositeMPOTensor(t.A')
end

function Base.adjoint(ts::Vector{CompositeMPOTensor})
    return convert(Vector{AdjointCompositeMPOTensor},[AdjointCompositeMPOTensor(t.A') for t in ts])
end

function Base.adjoint(t::AdjointCompositeMPOTensor)
    return CompositeMPOTensor(t.A')
end

function Base.adjoint(ts::Vector{AdjointCompositeMPOTensor})
    return convert(Vector{CompositeMPOTensor},[CompositeMPOTensor(t.A') for t in ts])
end

function Base.:*(A::CompositeMPOTensor{2,6}, B::AdjointCompositeMPOTensor{2,6})
    return  @tensor A.A[1,2,3,4,5,6] * B.A[4,5,6,1,2,3]
end

function Base.:*(A::DenseMPOTensor{4}, B::AdjointMPOTensor{4})
    return  @tensor A.A[1,2,3,4] * B.A[3,4,1,2]
end

function Base.:*(α::Number, A::CompositeMPOTensor)
    return  CompositeMPOTensor(α*A.A)
end

function Base.:*(α::Number, A::DenseMPOTensor)
    return  DenseMPOTensor(α*A.A)
end

function Base.:/(A::AbstractMPOTensor, α::Number)
    return  (1/α) * A
end

function Base.:+(A::CompositeMPOTensor{N₁, R₁}, B::CompositeMPOTensor{N₂, R₂}) where {N₁, N₂, R₁, R₂}
    @assert N₁ == N₂ && R₁ == R₂
    return CompositeMPOTensor(A.A + B.A)
end

function Base.:+(A::DenseMPOTensor{R₁}, B::DenseMPOTensor{R₂}) where {R₁, R₂}
    @assert R₁ == R₂
    return DenseMPOTensor(A.A + B.A)
end

function Base.:-(A::AbstractMPOTensor, B::AbstractMPOTensor)
    return A + (-1) * B
end

