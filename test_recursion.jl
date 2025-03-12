include("recursion.jl")
include("numbering.jl")
include("state_functions.jl")

# testing grounds for parallel_numbering

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)
composite  = viscous1, powerlaw

c1 = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

c2 = let
    # viscous -- parallel
    #               |  
    #     parallel --- viscous
    #         |  
    #      viscous
    #         |  
    #      viscous
    p1 = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

c3 = let
    # viscous -- parallel ------------- parallel
    #               |                       |
    #     parallel --- viscous    parallel --- viscous
    #         |                     |  
    #      viscous               viscous
    #         |                     |  
    #      viscous               viscous
    p1 = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p, p)
end

c4 = let
    # viscous -- parallel
    #               |
    #     parallel --- parallel
    #         |          |  
    #      viscous    viscous
    #         |          |  
    #      viscous    viscous
    p1 = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, p1)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, p)
end

@testset "Parallel numbering" begin
    @test parallel_numbering(c1) == ((1, ()),)
    @test parallel_numbering(c2) == ((1, ((2,),)),)
    @test parallel_numbering(c3) == ((1, ((2,),)), (3, ((4,),)))
    @test parallel_numbering(c4) == ((1, ((2,), (3,))),)
end

@testset "Functions mapping" begin
    # global functions for the series elements
    @test global_functions_numbering(c1) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
    @test global_functions_numbering(c2) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
    @test global_functions_numbering(c3) == (GlobalSeriesEquation{2, typeof(compute_strain_rate)}(1, (3, 4), compute_strain_rate),)
    @test global_functions_numbering(c4) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
    
    # mappings of the global functions of the parallel elements
    @test parallel_functions_numbering(c1) == (LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),)
    @test parallel_functions_numbering(c2) == (
        LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),
        LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate)
    )
    @test parallel_functions_numbering(c3) == (
        LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),
        LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(3, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(4, compute_strain_rate)
    )
    @test parallel_functions_numbering(c4) ==(
        LocalParallelEquation{typeof(compute_strain_rate)}(1, compute_strain_rate),
        LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate),
        LocalParallelEquation{typeof(compute_strain_rate)}(3, compute_strain_rate)
    )
end