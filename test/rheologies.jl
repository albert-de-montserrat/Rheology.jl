using Rheology


viscous1    = LinearViscosity(1e20)
viscous2    = LinearViscosity(5e19)
viscousbulk = BulkViscosity(1e18)
powerlaw    = PowerLawViscosity(5e19, 3)
drucker     = DruckerPrager(1e6, 10.0, 0.0)
elastic     = Elasticity(1e10, 1e12)
elasticbulk = BulkElasticity(1e10)
elasticinc  = IncompressibleElasticity(1e10)
LTP         = LTPViscosity(6.2e-13, 76, 1.8e9, 3.4e9)
diffusion   = DiffusionCreep(1, 1, 1, 1.5e-3, 1, 1, 1)
dislocation = DislocationCreep(3.5, 1, 1.1e-16, 1, 1, 1)

