function vrange(beginvec::Union{Vector,Tuple},endvec::Union{Vector,Tuple};step::Int64 = 100)
    return hcat([collect(beginvec .+ (endvec .- beginvec) .* t)  for t in range(0,1,step)]...)
end

function vrange(ipath::Matrix;eachstep::Number = 100)
    finalpath = ipath[:,1]

    for ii in 1:size(ipath)[2]-1
        finalpath = hcat(finalpath,vrange(ipath[:,ii],ipath[:,ii+1];step=eachstep+1)[:,2:end])
    end

    return finalpath
end

function pathlength(finalpath::Matrix)
    return cumsum(norm.(eachcol(hcat([0.0,0.0],diff(finalpath,dims = 2)))))
end


function kdivide(kvecpath::Matrix,groupn::Int64)
    nperg = div(size(kvecpath)[2] - 1,groupn)
    kg = []
    for ii in 1:groupn-1
        push!(kg,kvecpath[:,(ii-1)*nperg .+ (1:nperg)])
    end
    push!(kg,kvecpath[:,end-nperg:end])

    return kg
end

function kdivide(kr::Vector,groupn::Int64)
    nperg = div(length(kr) - 1,groupn)
    kg = []
    for ii in 1:groupn-1
        push!(kg,kr[(ii-1)*nperg .+ (1:nperg)])
    end
    push!(kg,kr[end-nperg:end])
    return kg
end

function diagm(dg::Vector{T}) where T
    L = length(dg)
    mat = zeros(T,L,L)
    for (dgi,dge) in enumerate(dg)
        mat[dgi,dgi] = dge
    end
    return mat
end

function diagm(pair::Pair{Int64, Vector{T}}) where T
    L = length(pair[2]) + abs(pair[1])
    mat = zeros(T,L,L)
    if pair[1] > 0
        for (ii,ie) in enumerate(pair[2])
            mat[ii,ii+pair[1]] = ie
        end
    elseif pair[1] < 0
        for (ii,ie) in enumerate(pair[2])
            mat[ii-pair[1],ii] = ie
        end
    else
        mat = diagm(pair[2])
    end
    
    return mat
end

FiniteLattices.coordinate(a::Union{Matrix,Vector};basis = KBASIS2) = basism(basis)*a

function kbasis3(basis::Vector)
    basis = collect.(basis)
    V = dot(basis[1],cross(basis[2],basis[3]))
    b1 = cross(basis[1],basis[2])*2*pi/V
    b2 = cross(basis[2],basis[3])*2*pi/V
    b3 = cross(basis[3],basis[1])*2*pi/V
    return Tuple.([b1,b2,b3])
end

function kbasis2(basis::Vector)
    kbasis = kbasis3(basis)
    kbasis2 = []
    for kvec in kbasis
        kvec[1] != kvec[2] && push!(kbasis2,kvec[1:2])
    end
    return kbasis2
end


# function FBZboundary!(ax::Axis,BDpoint::Matrix;
#     linewidth::Number = 2.0,
#     color::Symbol = :black,
#     showbasis::Bool = false,
#     arrowsize::Number = 0.2,
#     arrowwidth::Number = 2.0,
#     arrowcolor::Symbol = :blue,kwargs...
#     )
#     BDpoint = hcat(BDpoint,BDpoint[:,1])
#     boundary!(ax,BDpoint;linewidth = linewidth,color = color,kwargs...)
#     if showbasis
#         arrow0!(ax,0,0,KBASIS2[1]...;arrowsize = arrowsize,color = arrowcolor,linewidth = arrowwidth)
#         arrow0!(ax,0,0,KBASIS2[2]...;arrowsize = arrowsize,color = arrowcolor,linewidth = arrowwidth)
#     end

# end

# function boundary!(ax::Axis,BDpoint::Matrix;kwargs...)
#     boundary!(ax,collect(eachcol(BDpoint));kwargs...) 
# end

# function boundary!(ax::Axis,BDpoint::Vector;basis = KBASIS2,
#     linewidth::Number = 2.0,
#     color::Symbol = :black,kwargs...)
#     coord = [basism(basis)*collect(vec) for vec in BDpoint]
#     x = vcat([coord[ii][1] for ii in eachindex(coord)],coord[1][1])
#     y = vcat([coord[ii][2] for ii in eachindex(coord)],coord[1][2])
#     lines!(ax,x,y,linewidth = linewidth,color = color;kwargs...)
# end

# FBZboundary!(ax::Axis,BDpoint::Vector;kwargs...) = FBZboundary!(ax,hcat(collect.(BDpoint)...);kwargs...)


basism(basis::Vector) = hcat(collect.(basis)...)

function isinside(target::Vector,boundary::Matrix;isboundary::Bool = false,tol::Float64 = 1e-8)
    boundaryc = collect.(eachcol(boundary))
    map(x -> push!(x,0),boundaryc)
    targetc = vcat(target,0)
    judge = Vector{Bool}(undef,length(boundaryc))
    judge[end] = true
    std = cross(targetc .- boundaryc[end],boundaryc[1] .- boundaryc[end])[3]
    checkfunc(x,y) = isboundary ? >=(x,y) : >(x,y)
    for i in 1:length(boundaryc)-1
        judge[i] = checkfunc(cross(targetc .- boundaryc[i],boundaryc[i+1] .- boundaryc[i])[3] * std, -tol)
    end
    return sum(judge) == length(boundaryc)
end

isinside(target::Vector,boundary::Vector;basis = KBASIS2,kwargs...) = isinside(target,coordinate(hcat(collect.(boundary)...);basis = basis);kwargs...)

v2m(lsv::Vector) = hcat(collect.(lsv)...)
