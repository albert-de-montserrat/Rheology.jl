# Test file to evaluate some concepts that B has in mind

using LinearAlgebra
using StaticArrays
using ForwardDiff
using Test

include("composite.jl")
include("rheology_types.jl")
include("state_functions.jl")
include("matrices.jl")
#include("others.jl")

include("kwargs.jl")

#=
# struc that holds serial elements
struct Series{N} <: AbstractRheology
    elements::NTuple{N,AbstractRheology}
    number::Tuple # number of state functions per element
    N_jac::Int64                        # 
end
function Series(args...)
    N_jac = length(args)    
    number = number_elements(args)
    return Series(args, number, N_jac)
end
get_unique_state_functions(composite::Series) = get_unique_state_functions(composite.elements, :series) 

# struc that holds parallel elements
struct Parallel{N} <: AbstractRheology
    elements::NTuple{N,AbstractRheology}
    number::Tuple
    N_jac::Int64                        # 
end
function Parallel(args...)
    N_jac = length(args)    
    number = number_elements(args)
    return Parallel(args, number, N_jac)
end
get_unique_state_functions(composite::Parallel) = get_unique_state_functions(composite.elements, :parallel) 


viscous  = LinearViscosity(1e22)
powerlaw = PowerLawViscosity(5e19, 3)
elastic  = Elasticity(1e10, 1e100) # im making up numbers
drucker  = DruckerPrager(1e6, 30, 0) # C, ϕ, ψ

#composite =  (viscous, drucker,) #, powerlaw, 


c1  = Series(viscous, powerlaw, drucker)
c2  = Series(viscous, powerlaw, drucker, drucker)
c3  = Series(viscous, Series(drucker,powerlaw), drucker, drucker)
p1  = Parallel(viscous, powerlaw, drucker)
c4  = Series(viscous, Parallel(drucker,powerlaw), drucker, drucker)


#comp1 = Series(viscous, p1, viscous)


vars = input_vars = (; ε = 1e-15, λ = 0) # input variables
args = (; τ = 1e2, P = 1e6, dt = 1e10) # we solve for this, initial guess
statefuns  = get_unique_state_functions(c1)

args_diff, args_nondiff = split_args(args, statefuns)

# few remarks:
# we now get the correct variable names
keys(args_diff)

# rhs of the system of eqs, initial guess
x = SA[values(args_diff)...]

# needs adjustments
#R = compute_residual(composite.elements, statefuns, vars, args)


# Get the correct number of state functions for this case
statefuns, statenums = series_state_functions(c2.elements, c2.number)

# this takes care of repeated elements that are not the standard series elements
args_diff = differentiable_kwargs(statefuns, statenums)
=#



# --
# testing SeriesModel and CompositeModel
viscous  = LinearViscosity(1e22)
powerlaw = PowerLawViscosity(5e19, 3)
elastic  = Elasticity(1e10, 1e100) # im making up numbers
drucker  = DruckerPrager(1e6, 30, 0) # C, ϕ, ψ

#=
s1 = SeriesModel(viscous, drucker, drucker)
p1 = ParallelModel(viscous, powerlaw)
s5 = SeriesModel(s1, p1)
c1 = CompositeModel(s5)
c0 = CompositeModel(s1)

p2 = ParallelModel(viscous, powerlaw)
s2 = SeriesModel(viscous, p2, elastic)
c2 = CompositeModel(s2)

p3 = ParallelModel(s2, viscous)
c3 = CompositeModel(p3)

# get the numbering of each of the elements
numel = number_elements(c0)  # numbering of each element

# Get unique state functions for this rheology
statefuns, statenums = series_state_functions(s1.children, numel)
args_diff = differentiable_kwargs(statefuns, statenums)

# do the same for a parallel case:
numel = number_elements1(p1) 
statefuns, statenums = parallel_state_functions(p1.siblings, numel)
args_diff = differentiable_kwargs(statefuns, statenums)

# WIP: combinations of series and parallel
#numel = number_elements1(c1) 
#statefuns, statenums = series_state_functions(c1.components, numel)
=#

#=
# Series model with 2 plastic elements and a parallel model
s1 = SeriesModel(viscous, drucker, drucker)
p1 = ParallelModel(viscous, powerlaw)
s  = SeriesModel(s1, p1)
numel = number_elements1(s)
statefuns, statenums = series_state_functions1(s, numel)
args_diff = differentiable_kwargs(statefuns, statenums)
@test statenums == (0,2,3,4)
@test keys(args_diff) == (:τ, :λ_2, :λ_3, :ε_4)

# same but specified differently
p1 = ParallelModel(viscous, powerlaw)
s  = SeriesModel(viscous, drucker, drucker, p1)
numel = number_elements1(s)
statefuns, statenums = series_state_functions1(s, numel)
args_diff = differentiable_kwargs(statefuns, statenums)
@test statenums == (0,2,3,4)
@test keys(args_diff) == (:τ, :λ_2, :λ_3, :ε_4)

# 2 parallel elements in series
p1 = ParallelModel(viscous, powerlaw)
p2 = ParallelModel(viscous, powerlaw)
s = SeriesModel(p1, p2)
numel = number_elements1(s)
statefuns, statenums = series_state_functions1(s, numel)
args_diff = differentiable_kwargs(statefuns, statenums)
@test  keys(args_diff) == (:τ, :ε_1, :ε_4)

# Series that has a parallel model with plasticity
p1 = ParallelModel(drucker, powerlaw)
s  = SeriesModel(viscous, drucker, drucker, p1)
numel = number_elements1(s)
statefuns, statenums = series_state_functions1(s, numel)
args_diff = differentiable_kwargs(statefuns, statenums)
@test statenums == (0,2,3,4,5)
@test keys(args_diff) == (:τ, :λ_2, :λ_3, :ε_4, :λ_5)

# Parallel model with viscous rheologies
p  = ParallelModel(viscous, powerlaw)
numel = number_elements1(p)
statefuns, statenums = parallel_state_functions1(p, numel)
args_diff = differentiable_kwargs(statefuns, statenums)
@test statenums == (0,)
@test keys(args_diff) == (:ε,)

# Parallel model with serial rheology rheologies
s1 = SeriesModel(viscous, powerlaw)
p  = ParallelModel(s1, powerlaw)
numel = number_elements1(p)
statefuns, statenums = parallel_state_functions1(p, numel)
args_diff = differentiable_kwargs(statefuns, statenums)
@test statenums == (0,1)
@test keys(args_diff) == (:ε,:τ_1)

# Parallel model with serial rheology rheologies - Broken
s1 = SeriesModel(viscous, ParallelModel(powerlaw, viscous))
p  = ParallelModel(s1, powerlaw)
numel = number_elements1(p)
statefuns, statenums = parallel_state_functions1(p, numel)
args_diff = differentiable_kwargs(statefuns, statenums)
@test statenums == (0,1)
@test keys(args_diff) == (:ε,:τ_1)
=#

# simple viscous + powerlaw
#p         = ParallelModel(viscous, elastic)
p         = ParallelModel(viscous, viscous)

#s         = SeriesModel(viscous, elastic, p)
s         = SeriesModel(viscous, p)
#s         = SeriesModel(viscous, ParallelModel(SeriesModel(viscous,viscous), viscous))

c         = CompositeModel(s)

# get differentiable and residual args with 
diff_args, res_args  = get_all_kwargs(s) 




#=
statefuns, statenums = series_state_functions(c.components)

ind = findall(statefuns .== compute_strain_rate .|| statefuns .== compute_volumetric_strain_rate)
statenumsvec = [statenums...]
statenumsvec[ind] .= 1 #c.components.n[1]

# find the ParallelElements - we do keep them

statenums = Tuple(statenumsvec)

differentiable_kwargs(statefuns, statenums)
=#




# The strainrate is required for this element


#=
# example 1, paragraph 6
p         = ParallelModel(viscous, powerlaw)
s         = SeriesModel(elastic, p)
c         = CompositeModel(s)
statefuns, statenums = series_state_functions(c.components)


# example 2, paragraph 6
s31       = SeriesModel(viscous, elastic)     
p3        = ParallelModel(s31, viscous)
s         = SeriesModel(viscous, elastic, p3)
c         = CompositeModel(s)
statefuns, statenums = series_state_functions(c.components)
=#


#numel   = number_elements1(c)

#statefuns, statenums = series_state_functions1(s, numel)
