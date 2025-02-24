using LinearAlgebra
using StaticArrays
using ForwardDiff
using DifferentiationInterface

using Test 

include("../mwe_visc_elastic.jl")

@testset "test_jacobians.jl" begin 
    viscous1   = LinearViscosity(5e19)
    viscous2   = LinearViscosity(1e20)
    viscous1_s = LinearViscosityStress(5e19)
    powerlaw   = PowerLawViscosity(5e19, 3)
    drucker    = DruckerPrager(1e6, 10.0, 0.0)
    elastic    = Elasticity(1e10, 1e12) # im making up numbers
    ncases     = 8
    cases      = [Symbol("case$i") for i in 1:ncases]

    for case in cases
        composite, vars, args_solve0, args_other, J_true = if case === :case1 
            composite  = viscous1, powerlaw
            ε          = 1e-15 
            τ          = 1e2
            vars       = (; ε = ε) # input variables
            args_solve = (; τ = τ) # we solve for this, initial guess
            args_other = (;) # other args that may be needed, non differentiable
            # analytical Jacobian
            J11        = 1/2/viscous1.η  
            J22        = (2 * powerlaw.η) ^(1/powerlaw.n) * ε^(1/powerlaw.n-1) / powerlaw.n
            J = SA[
                J11     1.0
                -1.0    J22
            ]
            composite, vars, args_solve, args_other, J

        elseif case === :case2
            composite  = viscous1, elastic
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2,   P = 1e6  ) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            # analytical Jacobian
            J11        = 1/2/viscous1.η + 1/2/elastic.G/args_other.dt  
            J22        = 1 / elastic.K /args_other.dt  
            J = SA[
                J11    0.0
                0.0    J22
            ]
            composite, vars, args_solve, args_other, J

        elseif case === :case3
            composite  = viscous1, elastic, powerlaw
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2,   P = 1e6  ) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            # analytical Jacobian
            J11        = 1/2/viscous1.η + 1/2/elastic.G/args_other.dt  
            J22        = 1 / elastic.K /args_other.dt  
            J33        = (2 * powerlaw.η) ^(1/powerlaw.n) * vars.ε^(1/powerlaw.n-1) / powerlaw.n
            J = SA[
                J11  0.0  1.0
                0.0  J22  0.0
                -1.0  0.0  J33
            ]

            composite, vars, args_solve, args_other, J

        elseif case === :case4
            composite  = viscous1, drucker
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2, ) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            # analytical Jacobian
            J11        = 1/2/viscous1.η
            J = SA[
                J11    1.0
                -1.0   -1.0
            ]
            composite, vars, args_solve, args_other, J

        elseif case === :case5
            composite  = viscous1, elastic, powerlaw, drucker
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2,   P = 1e6,) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            # analytical Jacobian
            J11        = 1/2/viscous1.η 
            J22        = 1 / elastic.K /args_other.dt  
            J33        = (2 * powerlaw.η) ^(1/powerlaw.n) * vars.ε^(1/powerlaw.n-1) / powerlaw.n
            J = SA[
                J11       0.0       1.0      0.0
                0.0       J22       0.0      1.0
                -1.0       0.0       J33      0.0
                -0.0      -1.0      -0.0     -1.0
            ]
            composite, vars, args_solve, args_other, J

        elseif case === :case6
            composite  = powerlaw, powerlaw
            vars       = (; ε  = 1e-15) # input variables
            args_solve = (; τ  = 1e2) # we solve for this, initial guess
            args_other = (; ) # other args that may be needed, non differentiable
            # analytical Jacobian
            # J11        = 1/2/viscous1.η  
            J22 = J33  = (2 * powerlaw.η) ^(1/powerlaw.n) * vars.ε^(1/powerlaw.n-1) / powerlaw.n
            J = SA[
                0.0  1.0  1.0
                -1.0  J22  0.0
                -1.0  0.0  J33
            ]
            composite, vars, args_solve, args_other, J

        elseif case === :case7
            composite  = viscous1, viscous2
            vars       = (; ε  = 1e-15) # input variables
            args_solve = (; τ  = 1e2) # we solve for this, initial guess
            args_other = (; ) # other args that may be needed, non differentiable
            # analytical Jacobian
            J11        = 1/2/viscous1.η + 1/2/viscous2.η
            J = @SMatrix [
                J11
            ]
            composite, vars, args_solve, args_other, J

        elseif case === :case8
            composite  = viscous1, viscous2, drucker, drucker, elastic, powerlaw
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2,   P = 1e6,) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable

            # analytical Jacobian
            J11        = 1/2/viscous1.η + 1/2/viscous2.η + 1/2/elastic.G/args_other.dt  
            J22        = 1 / elastic.K /args_other.dt  
            J55        = (2 * powerlaw.η) ^(1/powerlaw.n) * vars.ε^(1/powerlaw.n-1) / powerlaw.n
            J = SA[
                J11       0.0       1.0   1.0   0.0
                0.0       J22       0.0   0.0   1.0
                -1.0      -0.0      -1.0  -0.0  -0.0
                -1.0      -0.0      -0.0  -1.0  -0.0
                0.0      -1.0       0.0   0.0   J55
            ]
            composite, vars, args_solve, args_other, J
        end

        J = main(composite, vars, args_solve0, args_other) 
        @test J ≈ J_true
    end
end