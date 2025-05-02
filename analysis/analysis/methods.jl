

function centralize(data::Union{Vector,OrdinalRange,StepRangeLen})
    return [(data[i] + data[i+1]) / 2 for i in 1:(length(data) - 1)]
end

function DOS(E::Array;n::Int64=size(E)[1])
    secs = range(extrema(E)...,n)
    dos = zeros(Int64,n-1)
    lsE = centralize(secs)

    for i in eachindex(secs)[1:end-1]
        dos[i] = count(x -> secs[i] <= x <= secs[i+1], E)
    end

    return lsE,dos / n^2
end

function OccupCurve(dos::Vector)
    return cumsum(dos)
end


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

function Densen(a::Vector,N::Int64)
    return range(extrema(a)...,N)
end


function ObsMat1(lsObsDict::Vector{Dict},name::String)
    lsdict = map(x -> x[name],lsObsDict)
    Lt = length(lsdict)
    LLatt = length(lsdict[1])
    M = zeros(LLatt,Lt)

    for it in 1:Lt
        M[:,it] = [lsdict[it][(iLatt,)] for iLatt in 1:LLatt]
    end

    return M
end
