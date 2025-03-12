include("recursion.jl")
include("state_functions.jl")

# complete flatten a tuple
@inline superflatten(t::NTuple{N, Any}) where N = superflatten(first(t))..., superflatten(Base.tail(t))... 
@inline superflatten(::Tuple{}) = ()
@inline superflatten(x::Number) = (x,)

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
    # get all the global functions of the series element
    fns_series              = global_series_state_functions(c)
    # templeate for the numbering of the parallel elements global equations
    eqnum_parallel_template = parallel_numbering(c) |> superflatten
    # offset in case there are more than one global functions to solve for
    offset                  = length(eqnum_parallel_template) 
    # generate pairs between global parallel equations and their related solution vector element
    ntuple(Val(length(fns_series))) do i
        @inline
        ntuple(Val(length(eqnum_parallel_template))) do j
            @inline
            LocalParallelEquation(eqnum_parallel_template[j] + offset * (i - 1), fns_series[i])
        end
    end |> flatten
end

"""
    global_functions_numbering(c::SeriesModel)

Assigns a global numbering to the functions within a `SeriesModel` object.

# Arguments
- `c::SeriesModel`: The `SeriesModel` object whose functions will be numbered.
"""
function global_functions_numbering(c::SeriesModel)
    # get all the global functions of the series elements
    fns_series       = global_series_state_functions(c) |> flatten_repeated_functions
    ns               = length(fns_series)

    # get all the local functions of the series elements
    fns_series_local = local_series_functions(c)
    ns_local         = length(fns_series_local)

    # local functions related to the global functions
    nleafs = length(c.leafs)
    inds_to_local = ntuple(Val(nleafs)) do j
        j ≤ ns_local ? (j + ns,) : ()
    end

    # parallel equations numbering
    eqnum_parallel   = parallel_numbering(c)
    np               = length(eqnum_parallel)

    # equations offsets
    offset_parallel  = np * ns

    # generate pairs between global parallel equations and their related solution vector element
    ntuple(Val(length(fns_series))) do i
        @inline
        # i = 2
        # parallel equations related to this global function
        inds_to_parallel = ntuple(Val(np)) do j
            (ns_local + ns) + np * (i - 1) + j
            # j - 1 + i + offset_parallel + ns_local + ns * (i - 1)
        end
        inds_to_all_local = (inds_to_local[i]..., inds_to_parallel...)
        GlobalSeriesEquation(i, inds_to_all_local, fns_series[i])
    end 
end
# global_functions_numbering(c)


# testing grounds

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)

composite  = viscous1, powerlaw
c1 = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

c2 = let
    # viscous -- parallel
    #               |  
    #     parallel --- viscous
    #         |  
    #      viscous
    #         |  
    #      viscous
    p1 = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

c3 = let
    # viscous -- parallel ------------- parallel
    #               |                       |
    #     parallel --- viscous    parallel --- viscous
    #         |                     |  
    #      viscous               viscous
    #         |                     |  
    #      viscous               viscous
    p1 = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p, p)
end

c4 = let
    # viscous -- parallel
    #               |
    #     parallel --- parallel
    #         |          |  
    #      viscous    viscous
    #         |          |  
    #      viscous    viscous
    p1 = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, p1)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

c5 = let
    # viscous -- powerlaw -- parallel
    #                           |  
    #                  viscous --- viscous  
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, powerlaw, p)
end

c6 = let
    # viscous -- elastic -- parallel
    #                          |  
    #                 viscous --- viscous  
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, elastic, p)
end

@test parallel_functions_numbering(c1) == (LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),)

@test parallel_functions_numbering(c2) == (
    LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),
    LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate)
)

@test parallel_functions_numbering(c3) == (
    LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),
    LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate), 
    LocalParallelEquation{typeof(compute_strain_rate)}(3, compute_strain_rate), 
    LocalParallelEquation{typeof(compute_strain_rate)}(4, compute_strain_rate)
)

@test parallel_functions_numbering(c4) ==(
    LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),
    LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate),
    LocalParallelEquation{typeof(compute_strain_rate)}(3, compute_strain_rate)
)

@test parallel_functions_numbering(c5) == (LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),)

@test global_functions_numbering(c1) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
@test global_functions_numbering(c2) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
@test global_functions_numbering(c3) == (GlobalSeriesEquation{2, typeof(compute_strain_rate)}(1, (3, 4), compute_strain_rate),)
@test global_functions_numbering(c4) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
@test global_functions_numbering(c5) == (GlobalSeriesEquation{2, typeof(compute_strain_rate)}(1, (2, 3), compute_strain_rate),)
@test global_functions_numbering(c6)       == (
    GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (3,), compute_strain_rate),
    GlobalSeriesEquation{1, typeof(compute_volumetric_strain_rate)}(2, (4,), compute_volumetric_strain_rate)
)