function plotbond!(ax::Union{Axis,Axis3},pairs::Vector, values::Vector, shift::Vector;
    colormap = :seismic,
    alphas = ones(length(pairs)),
    linewidth = 10,resizelevel = 1,
    N = 20,
    directed = false,
    colorintensity = 1,
    alphaintensity = 1,
    colorlimit = extrema(values)
    )
    @assert isequal(length.([pairs,values])...)

    # values01 = (values .- minimum(values)) ./ - -(extrema(values)...)
    # values11 = (values .- +(extrema(values)...)/2) ./ - -(extrema(values)...)

    for (ind,(i, j)) in enumerate(pairs)

        x = map([i, j]) do i
            coordinate(Latt, i)[1]
        end
        y = map([i, j]) do i
            coordinate(Latt, i)[2]
        end
    
        z = map([i,j]) do i
            Latt[i][1] - 1
        end
    
        # if norm(coordinate(Latt,i) .- coordinate(Latt,j)) > sqrt(3) - 1e-5 
        #     y[findmin(y)[2]] = y[findmin(y)[2]] + Ly*shift[2]
        #     x[findmin(x)[2]] = x[findmin(x)[2]] + Ly*shift[1]
        # end

        x .+= shift[1]
        y .+= shift[2]
        if directed
            directedlines!(ax, resize(x, y, z, resizelevel)...;
            linewidth=linewidth .* abs(values[ind]),
            colormap = colormap,
            colorintensity = colorintensity * values[ind],
            colorrange = colorlimit,
            N = N,
            alpha = alphaintensity * alphas[ind],
            )
        else
            lines!(ax, resize(x, y, z, resizelevel)...;
            linewidth=linewidth .* abs(values[ind]),
            color = get(colorschemes[colormap],values[ind],colorlimit),
            alpha = alphas[ind]
            )
        end
    end
end


function norm(A::Union{Vector,Tuple})
    return collect(A) |> x -> sqrt(sum(x .* x'))
end

function resize(X::Vector,Y::Vector,p::Number)
    A,B = map(1:2) do i
        [X[i],Y[i]]
    end
    r = @. (A - B) / 2
    c = @. (A + B) / 2
    A1 = @. c + p * r
    B1 = @. c - p * r
    X1,Y1 = map(1:2) do i 
        [A1[i],B1[i]]
    end
    return X1,Y1
end

function resize(X::Vector,Y::Vector,Z::Vector,p::Number)
    A,B = map(1:2) do i
        [X[i],Y[i],Z[i]]
    end
    r = @. (A - B) / 2
    c = @. (A + B) / 2
    A1 = @. c + p * r
    B1 = @. c - p * r
    X1,Y1,Z1 = map(1:3) do i 
        [A1[i],B1[i]]
    end
    return X1,Y1,Z1
end
