struct SingleSite <: SweepScheme SingleSite() = new() end
struct DoubleSite <: SweepScheme DoubleSite() = new() end

# struct singlesite <: SingleSite singlesite() = new() end
# struct doublesite <: DoubleSite doublesite() = new() end
struct randSVD <: CBEscheme 
    Df::Int64
    randSVD(Df::Int64) = new(Df)
    randSVD() = new(NaN)
end
struct fullSVD <: CBEscheme
    fullSVD() = new()
end
struct dynamicSVD <: CBEscheme
    Df::Int64
    dynamicSVD() = new(NaN)
    dynamicSVD(Df::Int64) = new(Df)
end

# struct randSVD <: CBEscheme 
#     complexity::Function
#     function randSVD()
#         F(D,d,w) = D^2 * d * w
#         return new(F) 
#     end
# end
