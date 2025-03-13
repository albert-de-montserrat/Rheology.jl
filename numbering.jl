include("recursion.jl")
include("state_functions.jl")

# complete flatten a tuple
@inline superflatten(t::NTuple{N, Any}) where N = superflatten(first(t))..., superflatten(Base.tail(t))... 
@inline superflatten(::Tuple{})                 = ()
@inline superflatten(x)                         = (x,)

struct SeriesModelEquations{T1, T2}
    fns_series::T1
    fns_parallel::T2

    function SeriesModelEquations(c::SeriesModel)
        fns_series   = global_functions_numbering(c)
        fns_parallel = parallel_functions_numbering(c)
        T1           = typeof(fns_series)
        T2           = typeof(fns_parallel)
        new{T1, T2}(fns_series, fns_parallel)
    end
end

struct GlobalSeriesEquation{N, F}
    eqnum::Int64 # equation number in the solution vector
    eqnums_reduce::NTuple{N, Int64} # equation numbers of the parallel elements that needed to be added to the residual vector
    fn::F # function to be evaluated
end

struct LocalParallelEquation{F}
    eqnum::Int64 # equation number in the solution vector
    fn::F # function to be evaluated
end

"""
    parallel_functions_numbering(c::SeriesModel)

Given a `SeriesModel` object `c`, this function generates a tuple of pairs where each pair consists of the globall functions corresponding to every (nested or not) parallel element, and to which element of the solution vector they are related with.

# Arguments
- `c::SeriesModel`: The `SeriesModel` object for which the numbering and functions are to be generated.
"""
function parallel_functions_numbering(c::SeriesModel)
    # # get all the global functions of the series element
    # fns_series              = global_series_state_functions(c) |> correct_fns_series
    # get all the functions of the parallel element
    fns                     = parallel_state_functions(c) |> superflatten
    # templeate for the numbering of the parallel elements global equations
    eqnum_parallel_template = parallel_numbering(c) |> superflatten
    # offset in case there are more than one global functions to solve for
    offset                  = length(eqnum_parallel_template) 
    # generate pairs between global parallel equations and their related solution vector element
    ntuple(Val(length(fns))) do i
        @inline
        ntuple(Val(length(eqnum_parallel_template))) do j
            @inline
            LocalParallelEquation(eqnum_parallel_template[j] + offset * (i - 1), fns[i])
        end
    end |> flatten
end

parallel_functions_numbering(c3)

@inline correct_fns_series(::Tuple{}) = (compute_strain_rate,)
@inline correct_fns_series(x)         = x

function local_indicies(::Val{nleafs}, ::Val{ns}, ::Val{ns_local}) where {nleafs, ns, ns_local} 
    ntuple(Val(nleafs)) do j
        j ≤ ns_local ? (j + ns,) : ()
    end
end

local_indicies(::Val{0}, ::Val{ns}, ::Val{ns_local}) where {ns, ns_local} = tuple(())

"""
    global_functions_numbering(c::SeriesModel)

Assigns a global numbering to the functions within a `SeriesModel` object.

# Arguments
- `c::SeriesModel`: The `SeriesModel` object whose functions will be numbered.
"""
function global_functions_numbering(c::SeriesModel)
    # get all the global functions of the series elements
    fns_series       = global_series_state_functions(c) |> flatten_repeated_functions |> correct_fns_series
    ns               = length(fns_series)

    # get all the local functions of the series elements
    fns_series_local = local_series_functions(c)
    ns_local         = length(fns_series_local)

    # local functions related to the global functions
    nleafs                = length(c.leafs)
    inds_to_local         = local_indicies(Val(nleafs), Val(ns), Val(ns_local))

    # parallel equations numbering
    eqnum_parallel        = parallel_numbering(c)
    np                    = length(eqnum_parallel) # number of parallel elements
    # length of the parallel equations of every main
    # parallel element, shifted one position to the right
    offset_parallel_local = (0, ntuple(i -> length(superflatten(eqnum_parallel[1])) - 1, Val(np))...)

    # equations offsets
    offset_parallel  = np * ns

    # generate pairs between global parallel equations and their related solution vector element
    ntuple(Val(length(fns_series))) do i
        @inline
        # parallel equations related to this global function
        inds_to_parallel = ntuple(Val(np)) do j
            offset_parallel_local[j] + (ns_local + ns) + np * (i - 1) + j
            # j - 1 + i + offset_parallel + ns_local + ns * (i - 1)
        end
        # @show inds_to_parallel
        inds_to_all_local = (inds_to_local[i]..., inds_to_parallel...)
        GlobalSeriesEquation(i, inds_to_all_local, fns_series[i])
    end
end

