using ForwardDiff, StaticArrays

import Base.IteratorsMD.flatten

include("others.jl")
include("rheology_types.jl")
include("state_functions.jl")
include("kwargs.jl")
include("composite.jl")
include("recursion.jl")
include("equations.jl")

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)

c, x, vars, args, others = let
    # elastic - viscous -- parallel
    #                         |  
    #                viscous --- viscous  |
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(viscous1, viscous2)
    c      = SeriesModel(elastic, viscous1, p)
    vars   = (; ε = 1e-15, θ = 1e-20) # input variables (constant)
    args   = (; τ = 1e2, P = 1e6)     # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)            # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c, x, vars, args, others = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, p)
    vars   = (; ε = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(args)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c, x, vars, args, others = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    p      = ParallelModel(viscous1, viscous2)
    c      = SeriesModel(viscous1, p)
    vars   = (; ε = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end


c, x, vars, args, others = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, p)
    vars   = (; ε = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(args)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

compute_residual(c, x, vars, others)
ForwardDiff.jacobian(x -> compute_residual(c, x, vars, others), x)

# @b compute_residual($(c, x, vars, others)...)
@b ForwardDiff.jacobian(y -> compute_residual($c, y, $vars, $others), $x)