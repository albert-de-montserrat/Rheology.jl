using LinearAlgebra
using StaticArrays
using ForwardDiff
using DifferentiationInterface

abstract type AbstractRheology end
abstract type AbstractPlasticity <: AbstractRheology end # in case we need spacilization at some point

abstract type AbstractCompositeModel  end

include("../src/rheology_types.jl")
include("../src/state_functions.jl")
include("../src/kwargs.jl")
# include("../src/matrices.jl")
include("others.jl")
include("residual.jl")


@inline series_state_functions(::AbstractCompositeModel)= ()
@inline parallel_state_functions(::AbstractCompositeModel)= ()

struct CompositeModel{Nstrain, Nstress, T} <: AbstractCompositeModel
    components::T
end

struct SeriesModel{L, B} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    leafs::L     # horizontal stacking
    branches::B  # vertical stacking

    function SeriesModel(c::Vararg{Any, N}) where N
        leafs = series_leafs(c)
        branches = series_branches(c)
        new{typeof(leafs), typeof(branches)}(leafs, branches)
    end
end


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

Base.show(io::IO, ::SeriesModel) = print(io, "SeriesModel")

struct ParallelModel{L, B} <: AbstractCompositeModel # not 100% about the subtyping here, lets see
    leafs::L     # horizontal stacking
    branches::B  # vertical stacking

    function ParallelModel(c::Vararg{Any, N}) where N
        leafs    = parallel_leafs(c)
        branches = parallel_branches(c)
        new{typeof(leafs), typeof(branches)}(leafs, branches)
    end
end

Base.show(io::IO, ::ParallelModel) = print(io, "ParallelModel")

@inline series_leafs(c::NTuple{N, AbstractRheology}) where N = c
@inline series_leafs(c::AbstractRheology) = (c,)
@inline series_leafs(::ParallelModel) = ()
@inline series_leafs(::Tuple{}) = ()
@inline series_leafs(c::NTuple{N, Any}) where N = series_leafs(first(c))..., series_leafs(Base.tail(c))...

@inline parallel_leafs(c::NTuple{N, AbstractRheology}) where N = c
@inline parallel_leafs(c::AbstractRheology) = (c,)
@inline parallel_leafs(::SeriesModel) = ()
@inline parallel_leafs(::Tuple{}) = ()
@inline parallel_leafs(c::NTuple{N, Any}) where N = parallel_leafs(first(c))..., parallel_leafs(Base.tail(c))...

@inline series_branches(::NTuple{N, AbstractRheology}) where N = ()
@inline series_branches(::AbstractRheology) = ()
@inline series_branches(c::ParallelModel) = (c,)
@inline series_branches(::Tuple{}) = ()
@inline series_branches(c::NTuple{N, Any}) where N = series_branches(first(c))..., series_branches(Base.tail(c))...

@inline parallel_branches(::NTuple{N, AbstractRheology}) where N = ()
@inline parallel_branches(::AbstractRheology) = ()
@inline parallel_branches(c::SeriesModel) = (c,)
@inline parallel_branches(::Tuple{}) = ()
@inline parallel_branches(c::NTuple{N, Any}) where N = parallel_branches(first(c))..., parallel_branches(Base.tail(c))...

Base.size(c::Union{SeriesModel, ParallelModel}) = length(c.leafs), length(c.branches)

for fun in (:compute_stress, :compute_pressure)
    @eval _local_parallel_state_functions(::typeof($fun)) = ()
    @eval @inline _global_parallel_state_functions(fn::typeof($fun)) = (fn, )
end
@inline _local_parallel_state_functions(fn::F) where F<:Function = (fn,)

@generated function local_parallel_state_functions(funs::NTuple{N, Any}) where N
    quote
        @inline
        f = Base.@ntuple $N i -> _local_parallel_state_functions(@inbounds(funs[i]))
        Base.IteratorsMD.flatten(f)
    end
end

@inline _global_parallel_state_functions(::F) where {F<:Function} = ()

@generated function global_parallel_state_functions(funs::NTuple{N, Any}) where N
    quote
        @inline
        f = Base.@ntuple $N i -> _global_parallel_state_functions(@inbounds(funs[i]))
        Base.IteratorsMD.flatten(f)
    end
end

@inline series_state_functions(c::NTuple{N, ParallelModel}) where {N} = series_state_functions(first(c))..., series_state_functions(Base.tail(c))...
@inline series_state_functions(c::ParallelModel)                      = flatten_repeated_functions(parallel_state_functions(c.leafs))
@inline series_state_functions(::Tuple{})                             = ()

######################################################################
viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
viscous1_s = LinearViscosityStress(5e19)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12) # im making up numbers

composite  = viscous1, powerlaw
p = ParallelModel(viscous1, powerlaw)
c = SeriesModel(viscous1, p)
c.leafs  # all the guys in series
c.branches[1].leafs    # all the guys in the parallel element(s)
c.branches[1].branches # all the guys in the series within a parallel leaf

vars       = (; ε  = 1e-15,) # input variables
args_solve = (; τ  = 1e2,  ) # we solve for this, initial guess
args_other = (; ) # other args that may be needed, non differentiable


# #######################################################################
# # DEAL FIRST WITH THE SERIES PART
# #######################################################################

# struct Composite{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11}
#     composite::T1               # initial definition of the composite
#     composite_expanded::T2      # composite for every row of the Jacobian (i.e. the local composite for the local equations)
#     composite_global::T3        # composite for every row of the Jacobian where the global equations are solved
#     state_funs::T4              # state functions
#     unique_funs_global::T5      # unique functions for the global equations
#     subtractor_vars::T6         # constant values that need to be subtracted from the state functions (i.e. input variables)
#     inds_x_to_subtractor::T7    # mapping from the state functions to the subtractor
#     inds_args_to_x::T8          # mapping from the local functions to the respective reduction of the state functions
#     reduction_ind::T9           # indices that map the local functions to the respective reduction of the state functions
#     args_template::T10          # template for args to do pattern matching later
#     N_reductions::T11           # number of reductions (i.e. variables we solve for)
# end

# function Composite(c::SeriesModel, vars, args_solve0)
#     composite          = c.leafs   
#     funs_all           = series_state_functions(composite)
#     funs_global        = global_series_state_functions(funs_all)
#     # args_solve         = merge(differentiable_kwargs(funs_global), args_solve0)
#     args_solve         = differentiable_kwargs(funs_global)
#     unique_funs_global = flatten_repeated_functions(funs_global)
    
#     funs_local         = local_series_state_functions(funs_all)
#     unique_funs_local  = flatten_repeated_functions(funs_local)
#     args_local         = all_differentiable_kwargs(funs_local)

#     # N_reductions         = length(unique_funs_local)
#     # N_reductions         = length(unique_funs_global)
#     N_reductions         = length(args_solve)
#     state_reductions     = ntuple(i -> state_var_reduction, Val(N_reductions))
#     args_reduction       = ntuple(_ -> (), Val(N_reductions))
#     state_funs           = merge_funs(state_reductions, funs_local)

#     # indices that map the local functions to the respective reduction of the state functions
#     reduction_ind        = reduction_funs_args_indices(funs_local, unique_funs_local)

#     N                    = length(state_funs)
#     # N_reductions0        = min(N_reductions, length(vars))  # to be checked
#     subtractor_vars      = SVector{N}(i ≤ N_reductions ? values(vars)[i] : 0e0 for i in 1:N)

#     inds_args_to_x       = generate_indices_from_args_to_x(funs_local, reduction_ind, Val(N_reductions))

#     # mapping from the state functions to the subtractor
#     inds_x_to_subtractor = mapping_x_to_subtractor(state_funs, unique_funs_local)
#     # template for args to do pattern matching later
#     args_template       = tuple(args_reduction..., args_local...)

#     # need to expand the composite for the local equations
#     composite_expanded  = expand_series_composite(composite, funs_local, Val(N_reductions))
#     composite_global,   = split_series_composite(composite, unique_funs_global, funs_local)
    
#     Composite(
#         composite,
#         composite_expanded,
#         composite_global,
#         state_funs,
#         unique_funs_global,
#         subtractor_vars,
#         inds_x_to_subtractor,
#         inds_args_to_x,
#         reduction_ind,
#         args_template,
#         Val(N_reductions)
#     )
# end

# Composite(c, vars, args_solve)

@inline global_series_functions(c::SeriesModel) = series_state_functions(c.leafs) |> flatten_repeated_functions |> global_series_state_functions
@inline local_series_functions(c::SeriesModel)  = series_state_functions(c.leafs) |> flatten_repeated_functions |> local_series_state_functions

@inline global_parallel_functions(c::SeriesModel) = ntuple(i-> parallel_state_functions(c.branches[i].leafs) |> flatten_repeated_functions |> global_parallel_state_functions, Val(count_parallel_elements(c))) 
@inline local_parallel_functions(c::SeriesModel)  = ntuple(i-> parallel_state_functions(c.branches[i].leafs) |> flatten_repeated_functions |> local_parallel_state_functions, Val(count_parallel_elements(c))) 

@inline count_series_elements(c::SeriesModel)           = length(c.leafs)
@inline count_parallel_elements(c::SeriesModel)         = length(c.branches)
# count all the equations
@inline count_unique_series_functions(c::SeriesModel)   = length(series_state_functions(c.leafs) |> flatten_repeated_functions)
@inline count_unique_parallel_functions(c::SeriesModel) = ntuple(i-> length(parallel_state_functions(c.branches[i].leafs)  |> flatten_repeated_functions ), Val(count_parallel_elements(c))) 
@inline count_parallel_functions(c::SeriesModel)        = ntuple(i-> length(parallel_state_functions(c.branches[i].leafs)), Val(count_parallel_elements(c))) 
@inline count_equations(c::SeriesModel)                 = count_unique_series_functions(c) + sum(count_unique_parallel_functions(c))
@inline count_series_equations(c::SeriesModel)          = count_unique_series_functions(c)
@inline count_parallel_equations(c::SeriesModel)        = count_parallel_functions(c)

# count global the equations (those of the elements in series, i.e. those parameters for which we solve)
@inline count_global_functions(c::SeriesModel)          = global_series_functions(c) |> length
# count local equations (if any) related to the elements in series
@inline count_local_series_functions(c::SeriesModel)    = local_series_functions(c) |> length


# count global the equations for the parallel elements (i.e. counterparts of the global equations)
@inline function count_global_parallel_functions(c::SeriesModel)
    N = count_parallel_elements(c)
    ntuple(Val(N)) do i 
        parallel_state_functions(c.branches[i].leafs) |> flatten_repeated_functions |> global_parallel_state_functions |> length
    end        
end
# count local equations (if any) related to the elements in series
@inline function count_local_parallel_functions(c::SeriesModel)
    N = count_parallel_elements(c)
    ntuple(Val(N)) do i 
        parallel_state_functions(c.branches[i].leafs) |> flatten_repeated_functions |> local_parallel_state_functions |> length
    end        
end

function residual_length(c::SeriesModel)
    nseries_global   = count_global_functions(c)
    nseries_local    = count_local_series_functions(c)
    nparallel_global = count_global_parallel_functions(c)
    nparallel_local  = count_local_parallel_functions(c)
    return nseries_global + nseries_local + sum(nparallel_global) + sum(nparallel_local)
end

Nr          = residual_length(c)
Np          = count_parallel_elements(c)
Ns          = count_series_elements(c)
x           = @SVector zeros(Nr) # solution vector
Ns_global   = count_global_functions(c)
Np_global   = count_global_parallel_functions(c)

Ns_equations        = count_series_equations(c) # total number of eqs in the series part
Np_global_equations = count_unique_parallel_functions(c) # number of global eqs in the parallel part

struct FunctionsAndArgs{T1, T2}
    fns::T1  # functions corresponding to the series and parallel parts 
    keys::T2 # keys of the arguments for every equation in the residual

    function FunctionsAndArgs(c::SeriesModel)
        Np                 = count_parallel_elements(c)

        # functions corresponding to the series part
        fns_global_series = global_series_functions(c)
        fns_local_series  = local_series_functions(c)
        fns_series        = (fns_global_series..., fns_local_series...)
        args_series       = differentiable_kwargs(fns_series)
        
        # functions corresponding to the parallel part(s)
        fns_global_parallel = global_parallel_functions(c)
        fns_local_parallel  = local_parallel_functions(c)
        fns_parallel        = ntuple(Val(Np)) do i 
            (fns_global_parallel[i]..., fns_local_parallel[i]...)
        end
        args_parallel       = differentiable_kwargs.(fns_parallel)
        
        fns = (fns_series, fns_parallel)
        # args_keys_total = Base.IteratorsMD.flatten(keys.((args_series, args_parallel...)))
        args_keys_total = (args_series, args_parallel...)
        
        new{typeof(fns), typeof(args_keys_total)}(fns, args_keys_total)
    end
end

fns_args = FunctionsAndArgs(c)

struct EquationNumbering{Nr, Ns_equations, Np, Np_global_equations}
    series::NTuple{Ns_equations, Int}
    parallel::NTuple{Np, NTuple{Np_global_equations, Int}}

    function EquationNumbering(c::SeriesModel)
        Ns_equations        = count_series_equations(c) # total number of eqs in the series part
        Np                  = count_parallel_elements(c)
        Np_global_equations = count_unique_parallel_functions(c) # number of global eqs in the parallel part

        eqnum_series   = ntuple(i -> i, Val(Ns_equations)) 
        eqnum_parallel = ntuple(Val(Np)) do i 
            ntuple(Val(Np_global_equations[i])) do j
                j + Ns_equations
            end
        end
        # number of residuals
        Nr = 1 + length(eqnum_parallel)
        new{Nr, Ns_equations, Np, sum(Np_global_equations)}(eqnum_series, eqnum_parallel)
    end
end

count_residuals(::EquationNumbering{N}) where N = N

eqnum = EquationNumbering(c)
eqnum.series
eqnum.parallel

x_mapping_series = ntuple(Val(Ns_equations)) do i
    i, getindex.(eqnum.parallel, i)...
end
x_mapping_parallel = ntuple(Val(Np)) do i
    getindex.(eqnum.parallel, i)
end
# this is maps the index of the elements in x that are needed as arguments for every residual equation
x_mapping    = (x_mapping_series..., x_mapping_parallel...)

function evaluate_residual_series(c::SeriesModel, x, fns_args, eqnum)
    arg_kwargs   = first(fns_args.keys)
    fns          = fns_args.fns
    fns_series   = first(fns)
    # fns_parallel = Base.tail(fns)

    Ns          = count_series_elements(c)
    keys_series = keys(arg_kwargs)
    val_series  = ntuple(Val(Ns)) do i
        x[i]
    end
    kwarg_series0 = (; zip(keys_series, val_series)...)

    # generate kwargs that are passed into the state functions
    Ns_equations = count_series_equations(c) # total number of eqs in the series part
    kwargs_series = ntuple(Val(Ns_equations)) do i
        fn       = fns_series[i]
        fn_kwarg = differentiable_kwargs(fn)
        merge(fn_kwarg, kwarg_series0)
    end

    # generate kwargs that need to be reduced, these are the variables in x that come from either local functions or parallel elements
    Ns_global   = count_global_functions(c)
    vals_to_reduce_series = ntuple(Val(Ns_global)) do i
        # x[i + Ns_global]
        inds = getindex.(eqnum.parallel, 1)
        ntuple(Val(length(inds))) do j
            x[inds[j]]
        end
    end

    residual_series = ntuple(Val(Ns_equations)) do i
        fns_series[i](c.leafs[i], kwargs_series[i]) + sum(vals_to_reduce_series[i])
    end
end

evaluate_residual_series(c, x, fns_args, eqnum)

# function evaluate_residual_parallel(c::SeriesModel, x, fns_args, eqnum)
    arg_kwargs   = Base.tail(fns_args.keys)
    fns          = Base.tail(fns_args.fns)

    Ns            = count_series_elements(c)
    Np            = count_parallel_elements(c)
    keys_parallel = keys.(arg_kwargs)
    val_parallel  = ntuple(Val(Np)) do i
        shift0 = i > 1 ? (i - 1) * length(fns[i-1]) : 0
        shift  = Ns + shift0
        x[i + Ns] 
    end
    kwarg_parallel0 = ntuple(Val(Np)) do i
        (; zip(keys_parallel[i], val_parallel[i])...)
    end

    # generate kwargs that are passed into the state functions
    kwargs_parallel = ntuple(Val(Np)) do j
        Np_equations = sum(length.(fns[j]))
        kwarg_parallelⱼ = kwarg_parallel0[j]
        ntuple(Val(Np_equations)) do i
            fn       = Base.IteratorsMD.flatten(fns[i])
            fn_kwarg = differentiable_kwargs(fn)
            merge(fn_kwarg, kwarg_parallelⱼ)
        end
    end

    # generate kwargs that need to be reduced, these are the variables in x that come from either local functions or parallel elements
    Ns_global   = count_global_functions(c)
    vals_to_reduce_parallel = ntuple(Val(Ns_global)) do i
        # x[i + Ns_global]
        inds = getindex.(eqnum.parallel, 1)
        ntuple(Val(length(inds))) do j
            x[inds[j]]
        end
    end

    # residual_parallel = ntuple(Val(Ns_equations)) do i
    #     fns[i](c.leafs[i], kwargs_parallel[i]) + sum(vals_to_reduce_parallel[i])
    # end
# end

@code_warntype evaluate_residual_parallel(c, x, fns_args, eqnum)
