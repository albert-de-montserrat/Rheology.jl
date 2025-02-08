# Test file to evaluate some concepts that B has in mind

using LinearAlgebra
using StaticArrays
using ForwardDiff

include("rheology_types.jl")
include("state_functions.jl")
include("kwargs.jl")
include("matrices.jl")
#include("others.jl")

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


composite        = Series(viscous, powerlaw, drucker)
p1        = Parallel(viscous, powerlaw, drucker)

#comp1 = Series(viscous, p1, viscous)



vars = input_vars = (; ε = 1e-15, λ = 0) # input variables
args = (; τ = 1e2, P = 1e6, dt = 1e10) # we solve for this, initial guess
statefuns  = get_unique_state_functions(composite)

args_diff, args_nondiff = split_args(args, statefuns)

# few remarks:
# we now get the correct variable names
keys(args_diff)

# rhs of the system of eqs, initial guess
x = SA[values(args_diff)...]

# needs adjustments
#R = compute_residual(composite.elements, statefuns, vars, args)
