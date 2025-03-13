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

c0 = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(viscous1, viscous2)
    SeriesModel(viscous1, p)
end

c1 = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
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
    #      parallel
    #         |  
    # viscous - viscous
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

c5 = let
    # viscous -- powerlaw -- parallel
    #                           |  
    #                 parallel --- viscous
    #                    |
    #                 viscous
    #                    |
    #                 viscous
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, powerlaw, p)
end

c6 = let
    # viscous -- elastic -- parallel
    #                          |  
    #                parallel --- viscous  
    #                   |
    #                viscous
    #                   |
    #                viscous
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, elastic, p)
end

c7 = let
    # viscous -- drucker -- parallel
    #                          |  
    #                parallel --- viscous  
    #                   |
    #                viscous
    #                   |
    #                viscous
    s1 = SeriesModel(viscous1, viscous2)
    p  = ParallelModel(s1, viscous2)
    SeriesModel(viscous1, drucker, p)
end

c8 = let
    #       parallel
    #          |
    # viscous --- viscous  
    p  = ParallelModel(viscous1, viscous2)
    SeriesModel(p)
end

@testset "Parallel numbering" begin
    @test parallel_numbering(c0) == ((1, ),)
    @test parallel_numbering(c1) == ((1, ()),)
    @test parallel_numbering(c2) == ((1, ((2,),)),)
    @test parallel_numbering(c3) == ((1, ((2,),)), (3, ((4,),)))
    @test parallel_numbering(c4) == ((1, ((2,), (3,))),)
    @test parallel_numbering(c5) == ((1, ()),)
    @test parallel_numbering(c7) == ((1, ()),)
end

@testset "Parallel functions" begin
    @test parallel_state_functions(c0) == (((compute_stress,),),)
    @test parallel_state_functions(c1) == (((compute_stress,), ((compute_strain_rate,),)),)
    @test parallel_state_functions(c2) == (((compute_stress,), ((compute_strain_rate,), ((compute_stress,),))),)
    @test parallel_state_functions(c3) == (((compute_stress,), ((compute_strain_rate,), ((compute_stress,),))), ((compute_stress,), ((compute_strain_rate,), ((compute_stress,),))))
    @test parallel_state_functions(c4) == (((compute_stress,), (((compute_stress,),), ((compute_stress,),))),)
    @test parallel_state_functions(c5) == (((compute_stress,), ((compute_strain_rate,),)),)
    @test parallel_state_functions(c6) == (((compute_stress,), ((compute_strain_rate,),)),)
    @test parallel_state_functions(c7) == (((compute_stress,), ((compute_strain_rate,),)),)

    @test parallel_state_functions(c0.branches[1]) == ((compute_stress,),)
    @test parallel_state_functions(c1.branches[1]) == ((compute_stress,), ((compute_strain_rate,),))
    @test parallel_state_functions(c2.branches[1]) == ((compute_stress,), ((compute_strain_rate,), ((compute_stress,),)))
    @test parallel_state_functions(c3.branches[1]) == ((compute_stress,), ((compute_strain_rate,), ((compute_stress,),)))
    @test parallel_state_functions(c3.branches[2]) == ((compute_stress,), ((compute_strain_rate,), ((compute_stress,),)))
    @test parallel_state_functions(c4.branches[1]) == ((compute_stress,), (((compute_stress,),), ((compute_stress,),)))
    @test parallel_state_functions(c5.branches[1]) == ((compute_stress,), ((compute_strain_rate,),))
    @test parallel_state_functions(c6.branches[1]) == ((compute_stress,), ((compute_strain_rate,),))
    @test parallel_state_functions(c7.branches[1]) == ((compute_stress,), ((compute_strain_rate,),))
end

@testset "Functions mapping" begin
    # global functions for the series elements
    @test global_functions_numbering(c0) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
    @test global_functions_numbering(c1) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
    @test global_functions_numbering(c2) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
    @test global_functions_numbering(c3) == (GlobalSeriesEquation{2, typeof(compute_strain_rate)}(1, (2, 4), compute_strain_rate),)
    @test global_functions_numbering(c4) == (GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (2,), compute_strain_rate),)
    @test global_functions_numbering(c5) == (GlobalSeriesEquation{2, typeof(compute_strain_rate)}(1, (2, 3), compute_strain_rate),)
    @test global_functions_numbering(c6) == (
        GlobalSeriesEquation{1, typeof(compute_strain_rate)}(1, (3,), compute_strain_rate),
        GlobalSeriesEquation{1, typeof(compute_volumetric_strain_rate)}(2, (4,), compute_volumetric_strain_rate)
    )
    @test global_functions_numbering(c7) == (GlobalSeriesEquation{2, typeof(compute_strain_rate)}(1, (2, 3), compute_strain_rate),)

    # mappings of the global functions of the parallel elements
    @test parallel_functions_numbering(c0) == (LocalParallelEquation{typeof(compute_stress)}(1, compute_stress),)
    @test parallel_functions_numbering(c1) == (
        LocalParallelEquation{typeof(compute_stress)}(1, compute_stress), 
        LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate)
    )
    @test parallel_functions_numbering(c2) == (
        LocalParallelEquation{typeof(compute_stress)}(1, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(2, compute_stress), 
        LocalParallelEquation{typeof(compute_strain_rate)}(3, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(4, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_stress)}(5, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(6, compute_stress),
    )
    @test parallel_functions_numbering(c3) == (
        LocalParallelEquation{typeof(compute_stress)}(1, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(2, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(3, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(4, compute_stress), 
        LocalParallelEquation{typeof(compute_strain_rate)}(5, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(6, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(7, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(8, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_stress)}(9, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(10, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(11, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(12, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(13, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(14, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(15, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(16, compute_stress), 
        LocalParallelEquation{typeof(compute_strain_rate)}(17, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(18, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(19, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_strain_rate)}(20, compute_strain_rate), 
        LocalParallelEquation{typeof(compute_stress)}(21, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(22, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(23, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(24, compute_stress)
    )
    @test parallel_functions_numbering(c4) == (
        LocalParallelEquation{typeof(compute_stress)}(1, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(2, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(3, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(4, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(5, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(6, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(7, compute_stress),
        LocalParallelEquation{typeof(compute_stress)}(8, compute_stress), 
        LocalParallelEquation{typeof(compute_stress)}(9, compute_stress)
    )
    @test parallel_functions_numbering(c5) == (
        LocalParallelEquation{typeof(compute_stress)}(1, compute_stress), 
        LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate)
    )
    @test parallel_functions_numbering(c6) == (
        LocalParallelEquation{typeof(compute_stress)}(1, compute_stress), 
        LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate)
    )
    @test parallel_functions_numbering(c7) == (
        LocalParallelEquation{typeof(compute_stress)}(1, compute_stress), 
        LocalParallelEquation{typeof(compute_strain_rate)}(2, compute_strain_rate)
    )
end