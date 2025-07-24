function mul!(α::Number, x::InteractionTreeLeave{N₁}, y::InteractionTreeLeave{N₂}, z::Union{Nothing,InteractionTreeLeave} = nothing) where {N₁,N₂}
    @assert x.Z == y.Z
    
    A = collect(x.A)
    site = collect(x.site)
    name = collect(x.name)
    fermionic = collect(x.fermionic)
    countZ = 0

    j = 1
    while j ≤ N₂
        i = length(A)
        while i > 0 && site[i] > y.site[j]
            fermionic[i] && (countZ += 1)
            i -= 1
        end
        if site[i] == y.site[j]
            A[i] = A[i]*y.A[j]
            name[i] = string(name[i],y.name[j])
            fermionic[i] = fermionic[i] | y.fermionic[j]
        else
            insert!(A, i + 1, y.A[j])
            insert!(site, i + 1, y.site[j])
            insert!(name, i + 1, y.name[j])
            insert!(fermionic, i + 1, y.fermionic[j])
        end
        j += 1
    end

    A[end] *= Z^countZ
    @show countZ
        
    obj = InteractionTreeLeave(map(x -> Tuple(x), [A,site,name,fermionic])..., x.strength * y.strength * α, x.Z)
    return replace!(z,obj)
end


