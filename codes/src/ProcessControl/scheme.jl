struct SingleSite <: SweepScheme SingleSite() = new() end
struct DoubleSite <: SweepScheme DoubleSite() = new() end

# struct singlesite <: SingleSite singlesite() = new() end
# struct doublesite <: DoubleSite doublesite() = new() end
struct randSVD <: CBEscheme randSVD() = new() end
# struct randSVD <: CBEscheme 
#     complexity::Function
#     function randSVD()
#         F(D,d,w) = D^2 * d * w
#         return new(F) 
#     end
# end
struct full <: CBEscheme full() = new() end
# struct randsvd <: randSVD randsvd() = new() end


