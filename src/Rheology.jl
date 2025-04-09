module Rheology

using ForwardDiff, StaticArrays, LinearAlgebra

import Base.IteratorsMD.flatten

include("rheology_types.jl")

export AbstractRheology, LinearViscosity, BulkViscosity, LinearViscosityStress, PowerLawViscosity, DiffusionCreep, DislocationCreep, LTPViscosity, Elasticity, IncompressibleElasticity, BulkElasticity, DruckerPrager

include("state_functions.jl")


include("kwargs.jl")

include("composite.jl")
export AbstractCompositeModel, CompositeModel, SeriesModel, ParallelModel

# include("recursion.jl")

include("equations.jl")
export CompositeEquation, generate_equations, compute_residual

include("others.jl")

include("function_utils.jl")

end # module Rheology
