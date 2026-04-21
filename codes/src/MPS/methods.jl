"""
Generate rand DenseMPS for initial state.
"""
function randMPS(PhySpaces::Vector,AuxSpaces::Vector;
    type::Type = ComplexF64,tailSpace::ElementarySpace = trivial(PhySpaces[1]))
    @assert (L = length(PhySpaces)) == length(AuxSpaces)
    push!(AuxSpaces, tailSpace)
    tmp = [MPSTensor(TensorMap(randn, type, AuxSpaces[i] ⊗ PhySpaces[i], AuxSpaces[i+1])) for i in 1:L]

    obj = DenseMPS{L,type}(tmp)

    canonicalize!(obj, L)
    canonicalize!(obj, 1)
    normalize!(obj)

    return obj
end

function randMPS(PhySpace::IndexSpace,AuxSpaces::Vector;kwargs...)
    return randMPS([PhySpace for i in eachindex(AuxSpaces)],AuxSpaces;kwargs...)
end

function AdjointMPSTensor(func, A::MPSTensor{<:Number, 3})
    cdm,dm = space(A.A) |> x -> (codomain(x),domain(x))
    tmp = AdjointMPSTensor(func,dm,cdm)
    normalize!(tmp)
    return tmp
end

