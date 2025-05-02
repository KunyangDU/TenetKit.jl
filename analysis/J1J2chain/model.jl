

function plotchainbond!(ax::Union{Axis,Axis3},Latt::AbstractLattice,pairs::Vector, values::Vector, shift::Vector;kwargs...)
    stagger = repeat([[0,0],[0,sqrt(3)]],div(size(Latt),2))
    plotbond!(ax,Latt,pairs,values,shift,stagger;kwargs...)
end

norm(A::Union{Vector,Tuple}) = collect(A) |> x -> sqrt(sum(x .* x'))

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
