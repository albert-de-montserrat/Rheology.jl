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




# The strainrate is required for this element


args = (; τ = 1e2, P = 1e6, dt = 1e10) # we solve for this, initial guess
vars = (; ε = 1e-15) # input variables


# example 1, paragraph 6
elastic_i = IncompressibleElasticity(1e10);
viscous   = LinearViscosity(1e18);
powerlaw  = PowerLawViscosity(5e19, 3);
p         = ParallelModel(viscous, powerlaw);
s         = SeriesModel(elastic_i, p);
c         = CompositeModel(s);
diff_args, res_args  = get_all_kwargs(c.components) 

#=
# example 2, paragraph 6
s31       = SeriesModel(viscous, elastic_i);     
p3        = ParallelModel(s31, viscous);
s         = SeriesModel(viscous, elastic_i, p3);
c         = CompositeModel(s);
diff_args, res_args  = get_all_kwargs(c.components)
=#


# define a serial component consisting of rheologies that only have a compute_stress function defined:
viscous_s =  LinearViscosityStress(1e19);
viscous   =  LinearViscosity(1e19);
s         = SeriesModel(viscous_s, viscous_s, viscous_s);
statefuns, statenums, stateelements = series_state_functions(s)
get_all_kwargs(s)

s1        = SeriesModel(viscous, viscous, viscous);
#diff_args, res_args  = get_all_kwargs(s1)

#flatten_repeated_functions(statefuns)


#R = compute_residual(c.components.children, statefuns, vars, args)
#J = compute_jacobian(x, composite, statefuns, args_diff, args_nondiff)


