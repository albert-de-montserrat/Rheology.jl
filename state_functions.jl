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

### 

function parallel_state_functions(c::SeriesModel)
    (; branches) = c
    ntuple(Val(length(branches))) do i
        parallel_state_functions(branches[i])
    end
end

@inline function parallel_state_functions(c::ParallelModel)
    (; leafs, branches) = c

    nl = length(leafs)
    fns_leafs = ntuple(Val(nl)) do i
        parallel_state_functions(leafs[i])
    end |> flatten_repeated_functions

    nb = length(branches)
    fns_branches = ntuple(Val(nb)) do i
        series_state_functions(branches[i])
    end # |> flatten_repeated_functions
    fns_leafs..., fns_branches...
end

@inline function series_state_functions(c::SeriesModel)
    (; leafs, branches) = c

    nl = length(leafs)
    fns_leafs = ntuple(Val(nl)) do i
        series_state_functions(leafs[i])
    end |> flatten_repeated_functions

    nb = length(branches)
    fns_branches = ntuple(Val(nb)) do i
        parallel_state_functions(branches[i])
    end #|> flatten_repeated_functions
    fns_leafs..., fns_branches...
end


