

function TrivialSSFT(Latt::AbstractLattice,data::Dict,lstk,points = 1:size(Latt);kwargs...)
    L = length(points)
    LL =size(Latt)
    SxSx = zeros(LL,LL)
    SySy = zeros(LL,LL)
    SzSz = zeros(LL,LL)

    for i in 1:L, j in i+1:L
        pair = points[i],points[j]
        SxSx[pair...] = data["SxSx"][pair]
        SySy[pair...] = data["SySy"][pair]
        SzSz[pair...] = data["SzSz"][pair]
    end

    So = zeros(LL)
    So[points] .= 1
    SSonsite = get(kwargs,:SSonsite,1/4)
    FSxSx,FSySy,FSzSz = map(x -> LL/L*FT2(x .+= x' + diagm(So)*SSonsite,Latt,lstk),[SxSx,SySy,SzSz])
    return FSxSx,FSySy,FSzSz
end

