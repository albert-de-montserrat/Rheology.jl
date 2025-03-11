include("recursion.jl")
include("state_functions.jl")

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
    fns_series = global_series_state_functions(c)
    # templeate for the numbering of the parallel elements global equations
    eqnum_parallel_template = parallel_numbering(c) |> superflatten
    # offset in case there are more than one global functions to solve for
    offset = length(eqnum_parallel_template) 
    # generate pairs between global parallel equations and their related solution vector element
    ntuple(Val(length(fns_series))) do i
        @inline
        ntuple(Val(length(eqnum_parallel_template))) do j
            @inline
            LocalParallelEquation(eqnum_parallel_template[j] + offset * (i-1), fns_series[i])
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
    # get all the global functions of the series element
    fns_series     = global_series_state_functions(c)
    ns             = length(fns_series)
    eqnum_parallel = parallel_numbering(c)
    np             = length(eqnum_parallel)
    offset_global  = np

    # generate paris between global parallel equations and their related solution vector element
    ntuple(Val(length(fns_series))) do i
        @inline
        inds_to_parallel = ntuple(j -> j - 1 + i + offset_global + ns * (i-1), Val(np))
        GlobalSeriesEquation(i, inds_to_parallel, fns_series[i])
    end 
end

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

@test global_functions_numbering(c1)   == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
@test global_functions_numbering(c2)   == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
@test global_functions_numbering(c3)   == (GlobalSeriesEquation{2, typeof(compute_strain_rate)}(1, (3, 4), compute_strain_rate),)
@test global_functions_numbering(c4)   == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)