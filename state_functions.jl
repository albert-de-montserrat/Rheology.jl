for fun in (:compute_strain_rate, :compute_volumetric_strain_rate)
    @eval @inline _local_series_state_functions(::typeof($fun)) = ()
    @eval @inline _global_series_state_functions(fn::typeof($fun)) = (fn, )
end
@inline _local_series_state_functions(fn::F) where F<:Function = (fn,)

@generated function local_series_state_functions(funs::NTuple{N, Any}) where N
    quote
        @inline
        f = Base.@ntuple $N i -> _local_series_state_functions(@inbounds(funs[i]))
        Base.IteratorsMD.flatten(f)
    end
end

@inline _global_series_state_functions(::F) where {F<:Function} = ()

@generated function global_series_state_functions(funs::NTuple{N, Any}) where N
    quote
        @inline
        f = Base.@ntuple $N i -> _global_series_state_functions(@inbounds(funs[i]))
        Base.IteratorsMD.flatten(f)
    end
end

function global_series_state_functions(c::SeriesModel)
    fns = series_state_functions(c.leafs)
    return global_series_state_functions(fns)
end

@inline series_state_functions(c::NTuple{N, ParallelModel}) where {N} = series_state_functions(first(c))..., series_state_functions(Base.tail(c))...
@inline series_state_functions(::Tuple{})                             = ()
# @inline series_state_functions(c::ParallelModel)                      = flatten_repeated_functions(parallel_state_functions(c.leafs))

function local_series_state_functions(c::SeriesModel)
    fns_series = global_series_state_functions(c)
    local_series_state_functions(fns_series)
end

@inline global_series_functions(c::SeriesModel) = series_state_functions(c.leafs) |> flatten_repeated_functions |> global_series_state_functions
@inline local_series_functions(c::SeriesModel)  = series_state_functions(c.leafs) |> flatten_repeated_functions |> local_series_state_functions
