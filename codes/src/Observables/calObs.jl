function calObs!(Obs::InteractionGraph, obj::T;kwargs...) where T <: Union{DenseMPO,DenseMPS}

    isnothing(Obs.graph) && initialize!(Obs)
    setdefault!(Obs,obj)
    
    if get_num_threads_julia() ≤ 100
        to = _calObs_serial!(Obs,obj;kwargs...)
    else
        to = _calObs_threading!(Obs,obj;kwargs...)
    end
    
    show(to,title = "calObs!")
    print("\n")
    flush(stdout)

    get(kwargs,:destroy,false) && (Obs.node = nothing)

    return Obs.values
end

function _calObs_serial!(obs::InteractionGraph,obj::T;kwargs...) where T <: Union{DenseMPO,DenseMPS}
    to = TimerOutput()
    stack = Union{DirectedNode,ObservableWeight}[]
    data = obs.values
    push!(stack, obs.graph.source[1],obs.graph.sink[1])
    while !isempty(stack)
        @timeit to "pop!"    task = popat!(stack,1)           # LIFO: depth-first
        @timeit to "update!" ans = _update!(task, obj)
        if typeof(ans) <: Tuple
            name,site,value = ans
            !haskey(data,name) && (data[name] = Dict{Tuple,Number}())
            @assert !haskey(data[name],site) "Observable Overcounted!"
            data[name][site] = value
        else
            @timeit to "push!" push!(stack, ans...)
        end
    end
    return to
end