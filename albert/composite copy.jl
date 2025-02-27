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


#######################################################################
# DEAL FIRST WITH THE SERIES PART
#######################################################################

struct Composite{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11}
    composite::T1               # initial definition of the composite
    composite_expanded::T2      # composite for every row of the Jacobian (i.e. the local composite for the local equations)
    composite_global::T3        # composite for every row of the Jacobian where the global equations are solved
    state_funs::T4              # state functions
    unique_funs_global::T5      # unique functions for the global equations
    subtractor_vars::T6         # constant values that need to be subtracted from the state functions (i.e. input variables)
    inds_x_to_subtractor::T7    # mapping from the state functions to the subtractor
    inds_args_to_x::T8          # mapping from the local functions to the respective reduction of the state functions
    reduction_ind::T9           # indices that map the local functions to the respective reduction of the state functions
    args_template::T10          # template for args to do pattern matching later
    N_reductions::T11           # number of reductions (i.e. variables we solve for)
end

function Composite(c::SeriesModel, vars, args_solve0)
    composite          = c.leafs   
    funs_all           = series_state_functions(composite)
    funs_global        = global_series_state_functions(funs_all)
    # args_solve         = merge(differentiable_kwargs(funs_global), args_solve0)
    args_solve         = differentiable_kwargs(funs_global)
    unique_funs_global = flatten_repeated_functions(funs_global)
    
    funs_local         = local_series_state_functions(funs_all)
    unique_funs_local  = flatten_repeated_functions(funs_local)
    args_local         = all_differentiable_kwargs(funs_local)

    # N_reductions         = length(unique_funs_local)
    # N_reductions         = length(unique_funs_global)
    N_reductions         = length(args_solve)
    state_reductions     = ntuple(i -> state_var_reduction, Val(N_reductions))
    args_reduction       = ntuple(_ -> (), Val(N_reductions))
    state_funs           = merge_funs(state_reductions, funs_local)

    # indices that map the local functions to the respective reduction of the state functions
    reduction_ind        = reduction_funs_args_indices(funs_local, unique_funs_local)

    N                    = length(state_funs)
    # N_reductions0        = min(N_reductions, length(vars))  # to be checked
    subtractor_vars      = SVector{N}(i ≤ N_reductions ? values(vars)[i] : 0e0 for i in 1:N)

    inds_args_to_x       = generate_indices_from_args_to_x(funs_local, reduction_ind, Val(N_reductions))

    # mapping from the state functions to the subtractor
    inds_x_to_subtractor = mapping_x_to_subtractor(state_funs, unique_funs_local)
    # template for args to do pattern matching later
    args_template       = tuple(args_reduction..., args_local...)

    # need to expand the composite for the local equations
    composite_expanded  = expand_series_composite(composite, funs_local, Val(N_reductions))
    composite_global,   = split_series_composite(composite, unique_funs_global, funs_local)
    
    Composite(
        composite,
        composite_expanded,
        composite_global,
        state_funs,
        unique_funs_global,
        subtractor_vars,
        inds_x_to_subtractor,
        inds_args_to_x,
        reduction_ind,
        args_template,
        Val(N_reductions)
    )
end

Composite(c, vars, args_solve)

composite          = c.branches[1].leafs
funs_all           = parallel_state_functions(composite)
funs_global        = global_parallel_state_functions(funs_all)
# args_solve         = merge(differentiable_kwargs(funs_global), args_solve0)
args_solve         = differentiable_kwargs(funs_global)

unique_funs_global = flatten_repeated_functions(funs_global)
args_global        = all_differentiable_kwargs(unique_funs_global)

args_global_aug    = ntuple(Val(length(args_global))) do i 
    merge(args_global[i], args_other)
end

funs_local         = local_parallel_state_functions(funs_all)
unique_funs_local  = flatten_repeated_functions(funs_local)
args_local         = all_differentiable_kwargs(funs_local)
vars_local        = ntuple(Val(length(args_local))) do i 
    k = keys(args_local[i])
    v = (first(k) ∈ keys(vars) ? vars[first(k)] : 0e0,)
    (; zip(k, v)...)
end
args_local_aug = ntuple(Val(length(args_local))) do i 
    merge(args_local[i], args_other, vars)
end
# N_reductions         = length(unique_funs_local)
# N_reductions         = length(unique_funs_global)
N_reductions         = length(args_solve)
state_reductions     = ntuple(i -> state_var_reduction, Val(N_reductions))
args_reduction       = ntuple(_ -> (), Val(N_reductions))
state_funs           = merge_funs(state_reductions, funs_local)

# indices that map the local functions to the respective reduction of the state functions
reduction_ind        = reduction_funs_args_indices(funs_local, unique_funs_local)

N                    = length(state_funs)
# N_reductions0        = min(N_reductions, length(vars))  # to be checked
subtractor_vars      = SVector{N}(i ≤ N_reductions ? values(vars)[i] : 0e0 for i in 1:N)

inds_args_to_x       = generate_indices_from_args_to_x(funs_local, reduction_ind, Val(N_reductions))

# mapping from the state functions to the subtractor
inds_x_to_subtractor = mapping_x_to_subtractor(state_funs, unique_funs_local)
# template for args to do pattern matching later
args_template        = tuple(args_reduction..., args_local...)

# need to expand the composite for the local equations
composite_expanded  = expand_parallel_composite(composite, funs_local, Val(N_reductions))
composite_global,   = split_parallel_composite(composite, unique_funs_global, funs_local)

local_x  = SA[Base.IteratorsMD.flatten(values.(vars_local))...]
global_x = SA[values(args_solve)...]
x        = SA[global_x..., local_x...]

# this are the values from the local components that need to be sumed in the state functions, i.e. in compute_strain_rate / compute_volumetric_strain_rate
args_state_reductions = generate_args_state_functions(reduction_ind, local_x, Val(N_reductions))

# 
args_all      = tuple(args_state_reductions..., args_local_aug...)

# R, J = value_and_jacobian(        
#     x -> eval_residual(x, composite_expanded, composite_global, state_funs, unique_funs_global, subtractor_vars, inds_x_to_subtractor, inds_args_to_x, args_template, args_all, args_solve, args_other),
#     AutoForwardDiff(), 
#     x
# )

args_solve        = (ε = 1e-15 / 2,)
subtractor        = generate_subtractor(x, inds_x_to_subtractor)
args_tmp          = generate_args_from_x(x, inds_args_to_x, args_template, args_all)
args_solve2       = merge(update_args2(args_solve, x), args_other)
subtractor_global = generate_subtractor_global(composite_global, state_funs, unique_funs_global, args_solve2)

eval_state_functions(state_funs, composite_expanded, args_tmp) - subtractor - subtractor_vars + subtractor_global

