"""
Generate rand DenseMPS for initial state.
"""
function randMPS(PhySpaces::Vector,AuxSpaces::Vector;
    type::Type = Float64,tailSpace::ElementarySpace = trivial(PhySpaces[1]),isdisk::Bool=IS_DISK[])
    @assert (L = length(PhySpaces)) == length(AuxSpaces)
    push!(AuxSpaces, tailSpace)
    tmp = Vector{MPSTensor}(undef,L)
    for i in 1:L
        tmp[i] = MPSTensor(ones,AuxSpaces[i] ⊗ PhySpaces[i],AuxSpaces[i+1])
    end

    obj = DenseMPS{L,type}(tmp;isdisk=isdisk)

    canonicalize!(obj, L)
    canonicalize!(obj, 1)
    normalize!(obj)

    return obj
end

function randMPS(PhySpace::IndexSpace,AuxSpaces::Vector;kwargs...)
    return randMPS([PhySpace for i in eachindex(AuxSpaces)],AuxSpaces;kwargs...)
end

function AdjointMPSTensor(func, A::MPSTensor{3})
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    tmp = AdjointMPSTensor(func,dm,cdm)
    normalize!(tmp)
    return tmp
end

