RBASIS3 = [[1/2,sqrt(3)/2,0.],[1/2,-sqrt(3)/2,0.],[0.,0.,1.]]
KBASIS2 = kbasis2(RBASIS3)
RBASIS2 = map(x -> x[1:2],RBASIS3[1:2])
FBZpoint = [[1/2,1/2],[1/2,-1/2],[-1/2,-1/2],[-1/2,1/2]]
MFBZpoint = [[1,1],[1,-1],[-1,-1],[-1,1]]

function TrivialSSFT(Latt::AbstractLattice,data::Dict,lstk,points = 1:size(Latt))
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
    FSxSx,FSySy,FSzSz = map(x -> LL/L*FT2(x .+= x' + diagm(So/4),Latt,lstk),[SxSx,SySy,SzSz])
    return FSxSx,FSySy,FSzSz
end
