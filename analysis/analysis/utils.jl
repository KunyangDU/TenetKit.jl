function easyinterp10(v,N=100)
    return 10. .^ (range(log10.(extrema(v))..., N))
end

