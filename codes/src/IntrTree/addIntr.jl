function addIntr!(Root::AbstractTreeNode,
    A::AbstractTensorMap,
    site::Int64,
    name::String,
    fermionic::Bool,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength ≈ 0 && return nothing
    addIntr1!(Root,A,site,name,fermionic,strength,Z)
end

function addIntr!(Root::AbstractTreeNode,
    A::NTuple{1,AbstractTensorMap},
    site::NTuple{1,Int64},
    name::NTuple{1,String},
    fermionic::NTuple{1,Bool},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength ≈ 0 && return nothing
    addIntr1!(Root,A[1],site[1],name[1],fermionic[1],strength,Z)
end

function addIntr!(Root::AbstractTreeNode,
    A::NTuple{2,AbstractTensorMap},
    site::NTuple{2,Int64},
    name::NTuple{2,String},
    fermionic::NTuple{2,Bool},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength ≈ 0 && return nothing
    addIntr2!(Root,A,site,name,fermionic,strength,Z)
end

function addIntr!(Tree::InteractionTree,
    A::AbstractTensorMap,
    site::Int64,
    name::String,
    fermionic::Bool,
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength ≈ 0 && return nothing
    addIntr1!(Tree.Root.children[1],A,site,name,fermionic,strength,Z)
end

function addIntr!(Tree::InteractionTree,
    A::NTuple{1,AbstractTensorMap},
    site::NTuple{1,Int64},
    name::NTuple{1,String},
    fermionic::NTuple{1,Bool},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength ≈ 0 && return nothing
    addIntr1!(Tree.Root.children[1],A[1],site[1],name[1],fermionic,strength,Z)
end

function addIntr!(Tree::InteractionTree,
    A::NTuple{2,AbstractTensorMap},
    site::NTuple{2,Int64},
    name::NTuple{2,String},
    fermionic::NTuple{2,Bool},
    strength::Number,
    Z::Union{Nothing,AbstractTensorMap})
    strength ≈ 0 && return nothing
    addIntr2!(Tree.Root.children[1],A,site,name,fermionic,strength,Z)
end

addIntr!(Root::InteractionTreeNode,Leave::InteractionTreeLeave{1}) = addIntr1!(Root,Leave)
addIntr!(Root::InteractionTreeNode,Leave::InteractionTreeLeave{2}) = addIntr2!(Root,Leave)

############# k ####################

# function addIntr!(Root::AbstractTreeNode,
#     A::AbstractTensorMap,
#     Latt::AbstractLattice,k::Vector,
#     name::String,
#     strength::Number,
#     string::Union{Nothing,AbstractTensorMap})
#     strength ≈ 0 && return nothing
#     L = size(Latt)
#     for site in 1:L
#         addIntr1!(Root,A,site,name,strength*exp(-1im*dot(k,coordinate(Latt,site))) / sqrt(L),string)
#     end
# end

# function addIntr!(Tree::InteractionTree,
#     A::AbstractTensorMap,
#     Latt::AbstractLattice,k::Vector,
#     name::String,
#     strength::Number,
#     string::Union{Nothing,AbstractTensorMap})
#     strength ≈ 0 && return nothing
#     addIntr!(Tree.Root.children[1],A,Latt,k,name,strength,string)
# end

# # c⁺ = exp(-1im)* ... 
# function addIntr!(Root::AbstractTreeNode,
#     A::Tuple,
#     Latt::AbstractLattice,k::Vector,
#     name::Tuple,
#     strength::Number,
#     Z::Union{Nothing,AbstractTensorMap})
#     L = size(Latt)
#     for i in 1:L, j in i+1:L, ind in 1:2 
#         addIntr2!(Root,A[ind],(i,j),name[ind],(-1)^(ind-1)*strength*exp((-1)^ind*1im*dot(k, coordinate(Latt,i) .- coordinate(Latt,j))) / L,Z)
#     end
# end
