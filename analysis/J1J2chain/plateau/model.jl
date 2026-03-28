function getCorrMat(Latt::AbstractLattice,data::Dict,dataonsite = nothing;
    selected_point = 1:size(Latt),
    independent_data = Dict(((i,),) => 0 for i in 1:size(Latt)))
    # L = length(selected_point)
    # A = zeros(L,L)
    # for i in 1:L,j in i+1:L
    #     A[i,j] = data[(selected_point[i],selected_point[j])]
    # end
    A = zeros(size(Latt),size(Latt))
    for i in selected_point, j in selected_point
        i >= j && continue
        A[i,j] = data[((i,j),)] - independent_data[((i,),)]*independent_data[((j,),)]
    end
    A = A + A'
    if !isnothing(dataonsite)
        if typeof(dataonsite) <: Dict
            for i in 1:L 
                A[i,i] = dataonsite[((i,),)] - independent_data[((i,),)]^2
            end
        elseif typeof(dataonsite) <: Vector
            A .+= diagm(dataonsite)
        elseif typeof(dataonsite) <: Number 
            # A .+= diagm(dataonsite*ones(L))
            tmp = zeros(size(Latt))
            tmp[selected_point] .= 1
            A .+= diagm((dataonsite .- [independent_data[((i,),)]^2 for i in 1:size(Latt)]) .* tmp)
        end
    end
    return A
end