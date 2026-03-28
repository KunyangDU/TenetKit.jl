
# example:
# - triangular
#     RBASIS3 = [(sqrt(3)/2,1/2,0.),(sqrt(3)/2,-1/2,0.),(0.,0.,1.)]
#     KBASIS2 = kbasis2(RBASIS3)
#     FBZpoint = [(1/6,1/3),(1/3,1/6),(1/6,-1/6),(-1/6,-1/3),(-1/3,-1/6),(-1/6,1/6)]
# - square
#     RBASIS3 = [(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]
#     KBASIS2 = kbasis2(RBASIS3)
#     FBZpoint = [(1/2,1/2),(1/2,-1/2),(-1/2,-1/2),(-1/2,1/2)]

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


function FBZboundary!(ax::Axis,BDpoint::Matrix;
    linewidth::Number = 2.0,
    color::Symbol = :black,
    showbasis::Bool = false,
    arrowsize::Number = 0.2,
    arrowwidth::Number = 2.0,
    arrowcolor::Symbol = :blue,kwargs...
    )
    BDpoint = hcat(BDpoint,BDpoint[:,1])
    boundary!(ax,BDpoint;linewidth = linewidth,color = color,kwargs...)
    if showbasis
        arrow0!(ax,0,0,KBASIS2[1]...;arrowsize = arrowsize,color = arrowcolor,linewidth = arrowwidth)
        arrow0!(ax,0,0,KBASIS2[2]...;arrowsize = arrowsize,color = arrowcolor,linewidth = arrowwidth)
    end

end

function boundary!(ax::Axis,BDpoint::Matrix;kwargs...)
    boundary!(ax,collect(eachcol(BDpoint));kwargs...) 
end

function boundary!(ax::Axis,BDpoint::Vector;basis = KBASIS2,
    shift = [0.,0.],
    linewidth::Number = 2.0,
    color::Symbol = :black,
    breathing = 1,kwargs...)
    coord = [basism(basis)*collect(vec) for vec in BDpoint]
    x = breathing*vcat([coord[ii][1] for ii in eachindex(coord)],coord[1][1]) .+ shift[1]
    y = breathing*vcat([coord[ii][2] for ii in eachindex(coord)],coord[1][2]) .+ shift[2]
    lines!(ax,x,y,linewidth = linewidth,color = color;kwargs...)
end

FBZboundary!(ax::Axis,BDpoint::Vector;kwargs...) = FBZboundary!(ax,hcat(collect.(BDpoint)...);kwargs...)


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
isinside(target::Tuple,boundary::Vector;basis = KBASIS2,kwargs...) = isinside(collect(target),boundary;kwargs...)


v2m(lsv::Vector) = hcat(collect.(lsv)...)

function vrange(lsv::Vector,N::Int64)
    Ls = [norm(lsv[i+1] .- lsv[i]) for i in 1:length(lsv)-1]
    Ns = Int.(round.(N*Ls/sum(Ls);digits=0))
    rnode = vcat(0,[sum(Ls[1:i]) for i in eachindex(Ls)])
    vpath = []
    rpath = []

    for (i,n) in enumerate(Ns)
        ts = range(0,1,n)
        push!(vpath,[lsv[i] + t*(lsv[i+1].-lsv[i]) for t in ts]...)
        push!(rpath,sum(Ls[1:i-1]) .+ ts*Ls[i]...)
    end
    return vpath,rpath,rnode
end
function orientate3(a::Vector,basis = [[0.,-1.],[-sqrt(3)/2,1/2],[sqrt(3)/2,1/2]])
    A = map(x -> dot(a,x),basis)
    return findfirst(x -> x == maximum(A),A),maximum(A)
end