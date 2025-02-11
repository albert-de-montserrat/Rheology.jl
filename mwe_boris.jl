# Test file to evaluate some concepts that B has in mind

using LinearAlgebra
using StaticArrays
using ForwardDiff

include("rheology_types.jl")
include("state_functions.jl")
include("matrices.jl")
#include("others.jl")

include("composite.jl")
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

# This now gives the expected result:
numel = number_elements1(s5)
statefuns, statenums = series_state_functions1(s5, numel)


numel = number_elements1(s1)
statefuns, statenums = series_state_functions1(s1, numel)
