"""
YCSqua() without L ≥ W check
"""
function iYCSqua(L::Int64, W::Int64, θ::Real = 0.0)
    e = ((1.0, 0.0), (0.0, 1.0))
    sites = [(x, y) for x in 1:L for y in 1:W]
    if iszero(θ)
         BC = PeriodicBoundaryCondition((0, W))
    else
         BC = TwistBoundaryCondition((0, W), θ)
    end
    return SquareLattice(e, sites, BC)
end


