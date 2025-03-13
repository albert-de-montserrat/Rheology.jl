include("numbering.jl")

# testing grounds

viscous1   = LinearViscosity(5e19)
viscous2   = LinearViscosity(1e20)
powerlaw   = PowerLawViscosity(5e19, 3)
drucker    = DruckerPrager(1e6, 10.0, 0.0)
elastic    = Elasticity(1e10, 1e12)

composite  = viscous1, powerlaw


c0, x0, vars0, args0, others0 = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(viscous1, viscous2)
    c      = SeriesModel(viscous1, p)
    vars   = (; ϵ = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end


c1, x1, vars1, args1, others1 = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, p)
    vars   = (; ϵ = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c2, x2, vars2, args2, others2 = let
    # viscous --- parallel
    #                |  
    #      parallel --- viscous
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
    
    vars   = (; ϵ = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c3, x3, vars3, args3, others3 = let
    # viscous -- parallel ------------- parallel
    #               |                       |
    #     parallel --- viscous    parallel --- viscous
    #         |                     |  
    #      viscous               viscous
    #         |                     |  
    #      viscous               viscous
    p1     = ParallelModel(viscous1, viscous2)
    s1     = SeriesModel(p1, viscous2)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, p, p)
    vars   = (; ϵ = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c4, x4, vars4, args4, others4 = let
    # viscous -- parallel
    #               |
    #     parallel --- parallel
    #         |          |  
    #      viscous    viscous
    #         |          |  
    #      viscous    viscous
    p1     = ParallelModel(viscous1, viscous2)
    s1     = SeriesModel(p1, p1)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, p)
    vars   = (; ϵ = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c5, x5, vars5, args5, others5 = let
    # viscous -- powerlaw -- parallel
    #                           |  
    #                 parallel --- viscous
    #                    |
    #                 viscous
    #                    |
    #                     viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, powerlaw, p)
    vars   = (; ϵ = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c6, x6, vars6, args6, others6 = let
    # viscous -- elastic -- parallel
    #                          |  
    #                parallel --- viscous  
    #                   |
    #                viscous
    #                   |
    #                viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, elastic, p)
    vars   = (;  ϵ = 1e-15, θ = 1e-20) # input variables (constant)
    args   = (;  τ = 1e2  , P = 1e6) # guess variables (we solve for these, differentiable)
    others = (; dt = 1e9)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

c7, x7, vars7, args7, others7 = let
    # viscous -- drucker -- parallel
    #                          |  
    #                parallel --- viscous  
    #                   |
    #                viscous
    #                   |
    #                    viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(viscous1, drucker, p)
    vars   = (; ϵ = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

function eval_series(c::SeriesModel, x::SVector, vars::NamedTuple, args::NamedTuple, others::NamedTuple)
    eqs         = SeriesModelEquations(c)
    leafs       = c.leafs
    nseries     = length(leafs)
    eqs_series  = eqs.fns_series
    neqs        = length(eqs_series)
    args_merged = merge(args, others)

    v = ntuple(Val(neqs)) do i
        @inline
        eq = eqs_series[i]
        fn = eq.fn
        
        # eval functions on the leafs
        v_series = ntuple(Val(nseries)) do j
            @inline
            fn(leafs[j], args_merged)
        end |> sum

        # reduce values coming from local/parallel equations
        inds = eq.eqnums_reduce
        v_local = ntuple(Val(length(inds))) do j
            @inline
            x[inds[j]]
        end |> sum

        v_series + v_local - vars[i]
    end
end


eval_series(c0, x0, vars0, args0, others0)
eval_series(c1, x1, vars1, args1, others1)
eval_series(c2, x2, vars2, args2, others2)
eval_series(c3, x3, vars3, args3, others3)
eval_series(c4, x4, vars4, args4, others4)
eval_series(c5, x5, vars5, args5, others5)
eval_series(c6, x6, vars6, args6, others6)
eval_series(c7, x7, vars7, args7, others7)


# @b eval_series($(c, x, vars, args, others)...)

