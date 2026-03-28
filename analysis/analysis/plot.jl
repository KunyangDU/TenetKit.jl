function plotbond!(ax::Union{Axis,Axis3},Latt::AbstractLattice,pairs::Vector, values::Vector, shift::Vector, stagger = [zeros(2) for _ in 1:size(Latt)];
    colormap = :seismic,
    alphas = ones(length(pairs)),
    linewidth = 10,resizelevel = 1,
    N = 20,
    directed = false,
    colorintensity = 1,
    alphaintensity = 1,
    colorlimit = extrema(values)
    )
    @assert isequal(length.([pairs,values])...)
    Lx,Ly = get_cellsize(Latt) 

    # values01 = (values .- minimum(values)) ./ - -(extrema(values)...)
    # values11 = (values .- +(extrema(values)...)/2) ./ - -(extrema(values)...)

    for (ind,(i, j)) in enumerate(pairs)

        x = map([i, j]) do i
            coordinate(Latt, i)[1] + stagger[i][1]
        end
        y = map([i, j]) do i
            coordinate(Latt, i)[2] + stagger[i][2]
        end
    
        z = map([i,j]) do i
            Latt[i][1] - 1
        end
    
        if abs((coordinate(Latt,i) .- coordinate(Latt,j))[2]) > Ly/2 - 1e-5 
            y[findmin(y)[2]] = y[findmin(y)[2]] + Ly*shift[2]
            x[findmin(x)[2]] = x[findmin(x)[2]] + Ly*shift[1]
        end
        if directed
            directedlines!(ax, resize(x, y, z, resizelevel)...;
            linewidth=linewidth .* abs(values[ind]),
            colormap = colormap,
            colorintensity = colorintensity * values[ind],
            colorrange = colorlimit,
            N = N,
            alpha = alphaintensity * alphas[ind],
            )
        else
            lines!(ax, resize(x, y, z, resizelevel)...;
            linewidth=linewidth .* abs(values[ind]),
            color = get(colorschemes[colormap],values[ind],colorlimit),
            alpha = alphas[ind]
            )
        end
    end
end

function plotLatt!(ax::Axis,Latt::AbstractLattice,shift::Vector;kwargs...)

    Lx,Ly = get_cellsize(Latt) 

    bond = get(kwargs, :bond, true)
    tplevel = get(kwargs, :tplevel, (1))
    site = get(kwargs, :site, false)
    selectedsite = get(kwargs,:selectedsite,1:size(Latt))
    sitelabel = get(kwargs, :sitelabel, true)
    sitesize = get(kwargs, :sitesize, 16 .* ones(length(selectedsite)))
    sitecolor = get(kwargs, :sitecolor, [:grey for _ in 1:length(selectedsite)])
    sitealpha = get(kwargs, :sitealpha, ones(length(selectedsite)))
    total_shift = get(kwargs, :total_shift, [0,0])
    
    linewidth = get(kwargs, :linewidth, 2)
    linecolor = get(kwargs, :linecolor, RGBf(0.5, 0.5, 0.5))


    if bond
        for level in tplevel
            # NN bond 
            for (i, j) in get(kwargs,:pairs,neighbor(Latt;level = level))

                    x = map([i, j]) do i
                        coordinate(Latt, i)[1] + total_shift[1]
                    end
                    y = map([i, j]) do i
                        coordinate(Latt, i)[2] + total_shift[2]
                    end

                    if abs((coordinate(Latt,i) .- coordinate(Latt,j))[2]) > Ly/2 - 1e-5 
                        x[findmin(x)[2]] = x[findmin(x)[2]] + shift[1]*Ly
                        y[findmin(y)[2]] = y[findmin(y)[2]] + shift[2]*Ly
                    end

                    lines!(ax, x, y;
                        linewidth=linewidth,
                        color=linecolor,
                    )
            end
        end
    end

    if site
        for (i,s) in enumerate(selectedsite)
                x, y = coordinate(Latt, s)

                CairoMakie.scatter!(ax, x + total_shift[1], y + total_shift[2];
                    markersize=sitesize[i],
                    color=sitecolor[i],
                    alpha = sitealpha[i])
        
                sitelabel && text!(ax, x + 0.05 + total_shift[1], y + 0.05 + total_shift[2]; text = "$s")
        end
    end
    
end

function FBZboundary!(ax::Axis,BDpoint::Matrix = FBZBD1;
    BDlinewidth::Number = 3.0,
    BDcolor::Symbol = :black,
    showbasis::Bool = false,
    arrowsize::Number = 0.2,
    arrowwidth::Number = 2.0,
    arrowcolor::Symbol = :blue,
    )
    coord = coordinate(BDpoint)
    x = coord[1,:]
    y = coord[2,:]

    lines!(ax,x,y,linewidth = BDlinewidth,color = BDcolor)

    if showbasis
        arrow0!(ax,0,0,KBASIS2[1]...;arrowsize = arrowsize,color = arrowcolor,linewidth = arrowwidth)
        arrow0!(ax,0,0,KBASIS2[2]...;arrowsize = arrowsize,color = arrowcolor,linewidth = arrowwidth)
    end

end

function arrow0!(ax::Axis,x, y, u, v; arrowsize=0.386, color=:black, transparency=1,linewidth = 1.2)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = arrowsize*nuv*v4, arrowsize*nuv*v5
    lines!(ax,[x,x+u], [y,y+v], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x+u,x+u-v5[1]], [y+v,y+v-v5[2]], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x+u,x+u-v4[1]], [y+v,y+v-v4[2]], color=(color,transparency),linewidth = linewidth)
end

function arrowc!(ax::Axis,x, y, u, v; kwargs...)
    arrow0!(ax,x-u/2,y-v/2,u,v;kwargs...)
end

function arrow2!(ax::Axis,x, y, u, v; arrowsize=0.386, color=:black, transparency=1,linewidth = 1.2)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = arrowsize*nuv*v4, arrowsize*nuv*v5
    lines!(ax,[x,x+u], [y,y+v], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x+u,x+u-v5[1]], [y+v,y+v-v5[2]], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x+u,x+u-v4[1]], [y+v,y+v-v4[2]], color=(color,transparency),linewidth = linewidth)
end

function resize(X::Vector,Y::Vector,p::Number)
    A,B = map(1:2) do i
        [X[i],Y[i]]
    end
    r = @. (A - B) / 2
    c = @. (A + B) / 2
    A1 = @. c + p * r
    B1 = @. c - p * r
    X1,Y1 = map(1:2) do i 
        [A1[i],B1[i]]
    end
    return X1,Y1
end

function resize(X::Vector,Y::Vector,Z::Vector,p::Number)
    A,B = map(1:2) do i
        [X[i],Y[i],Z[i]]
    end
    r = @. (A - B) / 2
    c = @. (A + B) / 2
    A1 = @. c + p * r
    B1 = @. c - p * r
    X1,Y1,Z1 = map(1:3) do i 
        [A1[i],B1[i]]
    end
    return X1,Y1,Z1
end

function directedlines!(ax::Union{Axis,Axis3},x,y,z;kwargs...)
    linewidth = get(kwargs,:linewidth,15)
    N = get(kwargs,:N, 20)
    colormap = get(kwargs,:colormap,:seismic)
    colorintensity = get(kwargs,:colorintensity,1)
    alpha = get(kwargs,:alpha,1)
    colorrange = get(kwargs,:colorrange,(-1,1))

    @assert abs(colorintensity) <= 1
    tmp = map(t -> range(t[1],(t[1] + t[2])/2,N), (x,y,z))
    #lscolor = get(colorschemes[colormap], colorintensity .* range(1,0,N),(-1,1))
    lslinewidth = linewidth .* range(1,0,N)
    for i in 1:N-1
        lines!(ax,map(t -> t[i:i+1], tmp)...;color = get(colorschemes[colormap], colorintensity,colorrange), linewidth = lslinewidth[i],alpha = alpha)
    end
end


function arrowz!(ax::Axis3,x,y,z, s; arrowsize=0.386, color=:black, transparency=1,linewidth = 1.2)
    u=0
    v=s
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u;v] / nuv,  [-v;u] / nuv
    v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2*(v4'*v2)*v2
    v4, v5 = arrowsize*nuv*v4, arrowsize*nuv*v5
    lines!(ax,[x,x],[y,y+u], [z,z+v], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x,x],[y+u,y+u-v5[1]], [z+v,z+v-v5[2]], color=(color,transparency),linewidth = linewidth)
    lines!(ax,[x,x],[y+u,y+u-v4[1]], [z+v,z+v-v4[2]], color=(color,transparency),linewidth = linewidth)
end

function polyHexagon!(ax::Axis, sites::Vector, colors::Vector;kwargs...)
    scale = get(kwargs, :scale, 1)
    alpha = get(kwargs, :alpha, 1)
    rot_mat = get(kwargs, :rot_mat, [1 0;0 1])
    for (i,center) in enumerate(sites)
        poly!(ax,
        Point2f[Tuple.([rot_mat*[cos(pi/3) sin(pi/3);-sin(pi/3) cos(pi/3)]^i*[0,1]*scale + center for i in 0:5])...], 
        color = (colors[i],alpha), strokewidth = 0)
    end
end

