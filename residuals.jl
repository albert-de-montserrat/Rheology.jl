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
    vars   = (; ε = 1e-15) # input variables (constant)
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
    c  = SeriesModel(viscous1, p)
    vars   = (; ε = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(args)..., # local  guess(es)
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
    p1 = ParallelModel(viscous1, viscous2)
    s1 = SeriesModel(p1, viscous2)
    p  = ParallelModel(s1, viscous2)
    c  = SeriesModel(viscous1, p, p)
    vars   = (; ε = 1e-15) # input variables (constant)
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
    vars   = (; ε = 1e-15) # input variables (constant)
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
    vars   = (; ε = 1e-15) # input variables (constant)
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
    vars   = (;  ε = 1e-15, θ = 1e-20) # input variables (constant)
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
    vars   = (; ε = 1e-15) # input variables (constant)
    args   = (; τ = 1e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end


@inline differentiable_kwargs(eqs::NTuple{N, GlobalSeriesEquation}) where N = merge(differentiable_kwargs(eqs[1]), differentiable_kwargs(Base.tail(eqs))...)
@inline differentiable_kwargs(eqs::NTuple{N, LocalParallelEquation}) where {N} = differentiable_kwargs(eqs[1]), differentiable_kwargs(Base.tail(eqs))...
@inline differentiable_kwargs(eqs::T) where {T<:Union{GlobalSeriesEquation,LocalParallelEquation}} = differentiable_kwargs(eqs.fn)
@inline differentiable_kwargs(::Tuple{})= ()

# @inline serialize(::Tuple{}) = ()

function serialize(c::AbstractCompositeModel)
    (; leafs, branches) = c
    x = ntuple(Val(length(branches))) do i
        @inline 
        serialize(branches[i])
    end |> flatten
    leafs, x...
end

function eval_series(c::SeriesModel, x::SVector, vars::NamedTuple, args::NamedTuple, others::NamedTuple)
    eqs         = SeriesModelEquations(c)
    leafs       = c.leafs
    nseries     = length(leafs)
    eqs_series  = eqs.fns_series
    neqs        = length(eqs_series)

    args_template = differentiable_kwargs(eqs_series)
    keys_series   = keys(args_template)
    val_series    = ntuple(i -> x[i], Val(nseries))
    args_tmp      = (; zip(keys_series, val_series)...)
    args_merged   = merge(args_tmp, vars, others)

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

function eval_parallel(c::SeriesModel, x::SVector, vars::NamedTuple, args::NamedTuple, others::NamedTuple)
    eqs          = SeriesModelEquations(c)
    (; branches) = c
    nbranches    = length(branches)
    eqs_parallel = eqs.fns_parallel
    nseries      = length(eqs.fns_series)
    neqs         = length(eqs_parallel)
    
    args_template = differentiable_kwargs(eqs_parallel)
    keys_parallel = keys.(args_template)
    val_parallel  = ntuple(i -> x[i + nseries], Val(neqs))
    args_merged   = ntuple(Val( length(val_parallel) )) do i 
        v = (; zip(keys_parallel[i], val_parallel[i])...)
        merge(v, others)
    end
    
    c_serial     = ntuple(i -> serialize(c.branches[i]), Val(nbranches)) # |> flatten
    np           = length(c_serial)
    
    counter = Ref(0)
    ntuple(Val(np)) do k
        cₖ = c_serial[k] # |> flatten
        ntuple(Val(length(cₖ))) do i
            @inline
            I  = counter[] += 1
            eq = eqs_parallel[I]
            fn = eq.fn
            cᵢ = cₖ[i] |> superflatten

            # eval functions
            v_parallel = ntuple(Val(length(cᵢ))) do j
                @inline
                fn(cᵢ[j], args_merged[i])
            end |> sum
    
            # # reduce values coming from local/parallel equations
            # inds = eq.eqnums_reduce
            # v_local = ntuple(Val(length(inds))) do j
            #     @inline
            #     x[inds[j]]
            # end |> sum

            # i dont like this, need to find a better way to do this
            v_local = i < length(cₖ) ? x[eq.eqnum + 1 + nseries] : 0.0

            v_parallel - x[eq.eqnum] + v_local
        end
    end |> flatten
end

function eval_residual(c::SeriesModel, x::SVector, vars::NamedTuple, args::NamedTuple, others::NamedTuple)
    r_series    =   eval_series(c, x, vars, args, others)
    r_parallel  = eval_parallel(c, x, vars, args, others)
    SA[r_series..., r_parallel...]
end

J0 = ForwardDiff.jacobian( x-> eval_residual(c0, x, vars0, args0, others0), x0)
J1 = ForwardDiff.jacobian( x-> eval_residual(c1, x, vars1, args1, others1), x1)
J2 = ForwardDiff.jacobian( x-> eval_residual(c2, x, vars2, args2, others2), x2)

@test J0 == SA[
    1.0e-20  1.0
   -1.0      3.0e20
]
@test J1 == SA[
    1.0e-20   1.0     0.0
   -1.0       2.0e20  1.0
    0.0      -1.0     1.5e-20
]

c, x, vars, args, others = c0, x0, vars0, args0, others0
c, x, vars, args, others = c2, x2, vars2, args2, others2

eval_parallel(c0, x0, vars0, args0, others0)
eval_parallel(c1, x1, vars1, args1, others1)
eval_parallel(c2, x2, vars2, args2, others2)
eval_parallel(c3, x3, vars3, args3, others3)
eval_parallel(c4, x4, vars4, args4, others4) # fails
eval_parallel(c5, x5, vars5, args5, others5) # fails
eval_parallel(c6, x6, vars6, args6, others6) # fails
eval_parallel(c7, x7, vars7, args7, others7) # fails

eval_series(c0, x0, vars0, args0, others0)
eval_series(c1, x1, vars1, args1, others1)
eval_series(c2, x2, vars2, args2, others2)
eval_series(c3, x3, vars3, args3, others3)
eval_series(c4, x4, vars4, args4, others4) # fails
eval_series(c5, x5, vars5, args5, others5) # fails
eval_series(c6, x6, vars6, args6, others6) # fails
eval_series(c7, x7, vars7, args7, others7) # fails

# @b eval_series($(c, x, vars, args, others)...)
