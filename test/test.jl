using Test


viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
viscous1_s = LinearViscosityStress(5e19)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12) # im making up numbers


@testset "test" begin
    for case in (:case1, :case2, :case3, :case4, :case5, :case6, :case7,)

        composite, vars, args_solve0, args_other = if case === :case1 
            composite  = viscous1, powerlaw
            vars       = (; ε  = 1e-15) # input variables
            args_solve = (; τ  = 1e2  ) # we solve for this, initial guess
            args_other = (;) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other

        elseif case === :case2
            composite  = viscous1, elastic
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2,   P = 1e6  ) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other

        elseif case === :case3
            composite  = viscous1, elastic, powerlaw
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2,   P = 1e6  ) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other

        elseif case === :case4
            composite  = viscous1, drucker
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2, ) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other

        elseif case === :case5
            composite  = viscous1, elastic, powerlaw, drucker
            vars       = (; ε  = 1e-15, θ = 1e-20) # input variables
            args_solve = (; τ  = 1e2,   P = 1e6,) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other

        elseif case === :case6
            composite  = powerlaw, powerlaw
            vars       = (; ε  = 1e-15) # input variables
            args_solve = (; τ  = 1e2) # we solve for this, initial guess
            args_other = (; dt = 1e10) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other

        elseif case === :case7
            composite  = viscous1, viscous2
            vars       = (; ε  = 1e-15) # input variables
            args_solve = (; τ  = 1e2) # we solve for this, initial guess
            args_other = (; ) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other

        elseif case === :case8
            composite  = viscous1, viscous2, drucker, drucker, elastic, powerlaw
            vars       = (; ε  = 1e-15) # input variables
            args_solve = (; τ  = 1e2, P = 10.0) # we solve for this, initial guess
            args_other = (; τ0  = 1.0, P0 = 1.0, dt = 1.0) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other


        elseif case === :case9
            composite  = viscous1, viscous2
            vars       = (; ε  = 1e-15) # input variables
            args_solve = (; τ  = 1e2,) # we solve for this, initial guess
            args_other = (; dt = 1.0) # other args that may be needed, non differentiable
            composite, vars, args_solve, args_other
        end
        success = try 
            main(composite, vars, args_solve0, args_other)
            "guet"
        catch
            "not guet"
        end
        if success == "not guet"
            println("case: ", case)
        end
        @test success == "guet"
    end
end