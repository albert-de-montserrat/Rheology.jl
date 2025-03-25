using ForwardDiff, StaticArrays, LinearAlgebra

import Base.IteratorsMD.flatten

include("others.jl")
include("rheology_types.jl")
include("state_functions.jl")
include("kwargs.jl")
include("composite.jl")
include("recursion.jl")
include("equations.jl")


viscous1    = LinearViscosity(5e19)
viscous2    = LinearViscosity(1e20)
powerlaw    = PowerLawViscosity(5e19, 3)
drucker     = DruckerPrager(1e6, 10.0, 0.0)
elastic     = Elasticity(1e10, 1e12)
LTP         = LTPViscosity(6.2e-13, 76, 1.8e9, 3.4e9)
diffusion   = DiffusionCreep(1, 1, 1, 1.5e-3, 1, 1, 1)
dislocation = DislocationCreep(3.5, 1, 1.1e-16, 1, 1, 1)

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
    args   = (; τ = 1e2)   # guess variables (we solve for these, differentiable)
    others = (;)           # other non-differentiable variables needed to evaluate the state functions

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
    #           parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
    s1     = SeriesModel(viscous1, viscous2)
    c      = ParallelModel(s1, viscous2)

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
    # viscous -- parallel    --      parallel
    #               |                   | 
    #      viscous --- viscous  viscous --- viscous  
    #         |  
    #      viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(s1, viscous2)
    p1     = ParallelModel(viscous1, viscous2)
    c      = SeriesModel(viscous1, p, p1)
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

# Arne's model 1
c, x, vars, args, others = let
    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
    s1     = SeriesModel(viscous1, viscous2, viscous1)
    p      = ParallelModel(s1, viscous2)
    c      = SeriesModel(elastic, p)
    vars   = (; ε = 1e-15, θ = 1e-20) # input variables (constant)
    args   = (; τ = 1e2,   P = 1e6)   # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)            # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(args)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

# Arne's model 2
c, x, vars, args, others = let
    viscous1    = LinearViscosity(1e15)
    viscous2    = LinearViscosity(1e15)
    elastic     = Elasticity(100e9, 1e12)
    LTP         = LTPViscosity(6.2e-13, 76, 1.8e9, 3.4e9)
    diffusion   = DiffusionCreep(1, 1, 1, 1.5e-3, 1, 1, 1)
    dislocation = DislocationCreep(3.5, 1, 1.1e-16, 1, 1, 1)

    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
    s1     = SeriesModel(diffusion, dislocation, LTP)
    p1     = ParallelModel(s1, viscous2)
    p2     = ParallelModel(elastic, viscous2)
    c      = SeriesModel(p1, p2)

    # p1     = ParallelModel(LTP, viscous1)
    # p2     = ParallelModel(elastic, viscous2)
    # c      = SeriesModel(p1, p2)
    
    # vars   = (; ε = 1e-12 * 2) # input variables (constant)
    args   = (; τ = 2e9)       # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)     # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(vars)..., # local  guess(es)
        0 .* values(args)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

# Arne's model 3
c, x, vars, args, others = let
    viscous1    = LinearViscosity(1e12)
    viscous2    = LinearViscosity(1e12)
    elastic     = Elasticity(100e9, 1e12)
    LTP         = LTPViscosity(6.2e-13, 76, 1.8e9, 3.4e9)
    diffusion   = DiffusionCreep(1, 1, 1, 1.5e-3, 1, 1, 1)
    dislocation = DislocationCreep(3.5, 1, 1.1e-16, 1, 1, 1)
    
    vars   = (; ε = 1e-12 * 2) # input variables (constant)
    args   = (; τ = 2e9)       # guess variables (we solve for these, differentiable)
    others = (; dt = 1e-2)     # other non-differentiable variables needed to evaluate the state functions

    # c      = SeriesModel(LTP)
    # x = SA[
    #     values(args)..., # global guess(es), solving for these
    # ]

    # p      = ParallelModel(LTP, viscous1)
    # c      = SeriesModel(p)
    # x = SA[
    #     values(args)..., # global guess(es), solving for these
    #     values(vars)..., # local  guess(es)
    # ]
    
    # s1     = SeriesModel(diffusion, dislocation, LTP)
    # p      = ParallelModel(s1, viscous1)
    # c      = SeriesModel(p)
    # x = SA[
    #     values(args)..., # global guess(es), solving for these
    #     1 .* values(vars)..., # local  guess(es)
    #     1 .* values(args)..., # global guess(es), solving for these
    # ]
    
    s1     = SeriesModel(diffusion, dislocation, LTP)
    p1     = ParallelModel(s1, viscous2)
    p2     = ParallelModel(elastic, viscous2)
    c      = SeriesModel(p1, p2)

    # vars   = (; ε = 1e-12 * 2) # input variables (constant)
    args   = (; τ = 2e9)       # guess variables (we solve for these, differentiable)
    others = (; dt = 1e1)     # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        0 .* values(vars)..., # local  guess(es)
        0 .* values(vars)..., # local  guess(es)
        0 .* values(args)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

function bt_line_search(Δx, J, x, r, composite, args, vars, others; α=1.0, ρ=0.5, c=1e-4, α_min=1e-8) where N

    eqs      = generate_equations(composite)
    args_all = generate_args_template(eqs, x, (;))

    perturbed_args = augment_args(args_all, α .* Δx)
    perturbed_r    = compute_residual(composite, x, vars, others)

    while sqrt(sum(perturbed_r.^2)) > sqrt(sum((r + (c * α * (J * Δx))).^2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_args = augment_args(args, α * Δx)
        perturbed_r    = compute_residual(composite, x, vars, others)
    end
    return α
end

function solve(c, x, vars, others)
    tol     = 1e-9
    itermax = 1e3
    it      = 0
    er      = Inf
    # Δx      = similar(x)
    while er > tol
        it += 1
        r  = compute_residual(c, x, vars, others)
        J  = ForwardDiff.jacobian(y -> compute_residual(c, y, vars, others), x)
        Δx = J \ r
        α  = bt_line_search(Δx, J, x, r, c, args, vars, others; α = 1)
        x -= α .* Δx
        er = norm(@. Δx/abs(x)) # norm(r)

        it > itermax && break
    end
    println("Iteration: $it, Error: $er, α = $α" )
    x
end

ε = exp10.(LinRange(log10(1e-15), log10(1e-8), 1000))
τ = similar(ε)
# args   = (; τ = 1e10)   # guess variables (we solve for these, differentiable)
# others = (; dt = 1e10) # other non-differentiable variables needed to evaluate the state functions
for i in eachindex(ε)
    vars = (; ε = ε[i]) # input variables (constant)
    τ[i] = solve(c, x, vars, others)[1]
end

f,ax,h    = lines(ε, τ)
ax.xlabel = L"\dot\varepsilon_{II}"
ax.ylabel = L"\tau_{II}"
f

# # compute_strain_rate(LTP, (; τ = 1e9) )
# # ForwardDiff.derivative(x -> compute_strain_rate(LTP, (; τ = x) ), 2e9)