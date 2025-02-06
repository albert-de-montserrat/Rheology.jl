using Test 

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

