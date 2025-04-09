using Test

include("rheologies.jl")

@testset "Test rheology structs" begin
    # Test for LinearViscosity
    @test viscous1.η == 1e20
    @test viscous2.η == 5e19

    # Test for BulkViscosity
    @test viscousbulk.χ == 1e18

    # Test for PowerLawViscosity
    @test powerlaw.η == 5e19
    @test powerlaw.n == 3

    # Test for DruckerPrager
    @test drucker.C == 1e6
    @test drucker.ϕ == 10.0
    @test drucker.ψ == 0.0

    # Test for Elasticity
    @test elastic.G == 1e10
    @test elastic.K == 1e12

    # Test for BulkElasticity
    @test elasticbulk.K == 1e10

    # Test for IncompressibleElasticity
    @test elasticinc.G == 1e10

    # Test for LTPViscosity
    @test LTP.Q  == 76
    @test LTP.ε0 == 6.2e-13
    @test LTP.σb == 1.8e9
    @test LTP.σr == 3.4e9

    # Test for DiffusionCreep
    @test diffusion.A == 0.0015
    @test diffusion.R == 1
    @test diffusion.n == 1
    @test diffusion.r == 1
    @test diffusion.E == 1
    @test diffusion.V == 1
    @test diffusion.p == 1

    # Test for DislocationCreep
    @test dislocation.A == 1.1e-16
    @test dislocation.R == 1
    @test dislocation.n == 3.5
    @test dislocation.r == 1
    @test dislocation.E == 1
    @test dislocation.V == 1
end