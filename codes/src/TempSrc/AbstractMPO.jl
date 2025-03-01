


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

        last_leaves = []
        last_roots = Root.children
        #idtensor =isometry((codomain(last_roots[1].children[1].Opr.Opri)[1] |> x -> (x,x))...)
        idtensor = getIdTensor(last_roots[1].children[1].Opr)
        
        last_inverse_root = 0
        next_inverse_root = 0
        
        for iL in 1:L

            if next_inverse_root == 0 && !isempty(findall(x -> isempty(x.children),vcat([lastroot.children for lastroot in last_roots]...)))
                next_inverse_root = 1
            end

            next_leaves = []
            leaves_inds = []

            next_roots = []
            roots_inds = []

            for (lastind,last_root) in enumerate(last_roots)
                for next_subtree in last_root.children
                    if isempty(next_subtree.children)
                        push!(next_leaves,next_subtree)
                        push!(leaves_inds,(lastind + last_inverse_root, next_inverse_root, length(next_leaves)))
                    else
                        push!(next_roots,next_subtree)
                        push!(roots_inds,(lastind + last_inverse_root, length(next_roots) + next_inverse_root, length(next_roots)))
                    end
                end
            end

            localMPOdims = length.((last_roots,next_roots)) .+ (last_inverse_root,next_inverse_root)
            localMPO = SparseMPOTensor(nothing,localMPOdims...)
            #localMPO .= DenseMPOTensor(0*idtensor)
            localMPO.m[1,1] = DenseMPOTensor(last_inverse_root*idtensor)


            for inds in leaves_inds
                localMPO.m[inds[1:2]...] = DenseMPOTensor(let 
                    localOpr = next_leaves[inds[3]].Opr.Opri
                    strength = next_leaves[inds[3]].Opr.strength
                    if isnan(strength)
                        localOpr
                    else
                        localOpr*strength
                    end
                end)
            end

            for inds in roots_inds
                localMPO.m[inds[1:2]...] = DenseMPOTensor(let 
                    localOpr = next_roots[inds[3]].Opr.Opri
                    strength = next_roots[inds[3]].Opr.strength
                    if isnan(strength)
                        localOpr
                    else
                        localOpr*strength
                    end
                end)
            end
            
            last_leaves = next_leaves
            last_roots = next_roots
            last_inverse_root = next_inverse_root

            tempMPO[iL] = localMPO
        end

        SparseMPO(tempMPO)
    end

    return MPO
end

function AutomataSparseMPO(Tree::InteractionTree,L::Int64 = treeheight(Tree.Root) - 1)
    return AutomataSparseMPO(Tree.Root,L)
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

