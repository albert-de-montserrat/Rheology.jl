using Test 

using LinearAlgebra
using StaticArrays
using ForwardDiff

include("../rheology_types.jl")
include("../state_functions.jl")
include("../composite.jl")
include("../matrices.jl")
include("../kwargs.jl")
include("../others.jl")

viscous  = LinearViscosity(1e18)
powerlaw = PowerLawViscosity(5e19, 3)
elastic  = Elasticity(1e10, 1e100) # im making up numbers
drucker  = DruckerPrager(1e6, 30, 0) # C, ϕ, ψ

p1       = ParallelModel(viscous, drucker)
s1       = SeriesModel(elastic, powerlaw, p1)
c1       = CompositeModel(s1)

@test number_strain_rate_components(c1) == 3
@test number_stress_components(c1)      == 2

s2       = SeriesModel(elastic, powerlaw)
p2       = ParallelModel(viscous, s2)
c2       = CompositeModel(p2)

@test number_strain_rate_components(c2) == 2
@test number_stress_components(c2)      == 2

s0 = SeriesModel(elastic, powerlaw, p1);

p1 = ParallelModel(viscous, elastic);
s1 = SeriesModel(elastic, powerlaw, p1);
c1 = CompositeModel(s1);
funs_series   = c1.components.funs
funs_parallel = c1.components.children[3].funs
all_funs = (funs_series..., funs_parallel)


p2 = ParallelModel(viscous, SeriesModel(elastic, viscous));
s2 = SeriesModel(elastic, powerlaw, p2);
c2 = CompositeModel(s2);
funs_s  = c2.components.funs;
funs_p  = c2.components.children[3].funs;
funs_s2 = c2.components.children[3].siblings[2].funs;
all_funs = (funs_s..., (funs_p, funs_s2))

