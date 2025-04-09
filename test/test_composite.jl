using Test 

include("rheologies.jl")

@testset "Test composite models" begin
    s1 = SeriesModel(viscous1, viscous2)

    @test s1.leafs    == (viscous1, viscous2)
    @test s1.branches == ()

    p1 = ParallelModel(drucker, elastic)
    s2 = SeriesModel(viscous1, p1)

    @test s2.leafs                == (viscous1, )
    @test s2.branches             == (p1, )
    @test s2.branches[1].leafs    == (drucker, elastic)
    @test s2.branches[1].branches == ()

    p2 = ParallelModel(viscous1, powerlaw)
    s3 = SeriesModel(p1, p2)

    @test s3.leafs                == ()
    @test s3.branches             == (p1, p2)
    @test s3.branches[1].leafs    == (drucker, elastic)
    @test s3.branches[1].branches == ()
    @test s3.branches[2].leafs    == (viscous1, powerlaw)
    @test s3.branches[2].branches == ()

    p3 = ParallelModel(viscous1, s1)
    s4 = SeriesModel(viscous1, p3)

    @test s4.leafs                         == (viscous1, )
    @test s4.branches                      == (p3, )
    @test s4.branches[1].leafs             == (viscous1, )
    @test s4.branches[1].branches          == (s1, )
    @test s4.branches[1].branches[1].leafs == s1.leafs
end
