function Base.getindex(obj::InteractionTunnel{L,<:ObservableOperator,N}, i::Int64) where {L,N}
    sites = map(x -> x.site, obj.A)
    idx = findfirst(x -> x == i, sites)
    if idx !== nothing
        @assert !obj.A[idx].isstring
        return obj.A[idx]
    end
    isnothing(obj.Z) && return ObservableOperator(i)
    return iseven(sum([(sites[j] > i && obj.fermionic[j]) ? 1 : 0 for j in 1:N])) ? ObservableOperator(i) : ObservableOperator(obj.Z, "Z", i,true)
end
