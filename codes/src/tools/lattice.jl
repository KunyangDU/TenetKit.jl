get_cellsize(Latt::CompositeLattice) = map(x -> maximum([Latt.subLatts[1].sites[ii][x] for ii in 1:div(size(Latt),length(Latt.subLatts))]),1:2)
get_cellsize(Latt::SimpleLattice) = map(x -> maximum([Latt.sites[ii][x] for ii in 1:size(Latt)]),1:2)


