using ForwardDiff, StaticArrays, LinearAlgebra
using GLMakie

import Base.IteratorsMD.flatten

include("rheology_types.jl")
include("state_functions.jl")
include("composite.jl")
include("kwargs.jl")
include("recursion.jl")
include("equations.jl")
include("others.jl")
include("../src/print_rheology.jl")

function bt_line_search(Δx, J, x, r, composite, vars, others; α = 1.0, ρ = 0.5, c = 1.0e-4, α_min = 1.0e-8)

    perturbed_x = @. x - α * Δx
    perturbed_r = compute_residual(composite, perturbed_x, vars, others)

    while sqrt(sum(perturbed_r .^ 2)) > sqrt(sum((r + (c * α * (J * Δx))) .^ 2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_x = @. x - α * Δx
        perturbed_r = compute_residual(composite, perturbed_x, vars, others)
    end
    return α
end

function solve(c, x, vars, others)
    tol = 1.0e-9
    itermax = 10.0e3
    it = 0
    er = Inf
    # Δx      = similar(x)
    local α
    while er > tol
        it += 1
        r = compute_residual(c, x, vars, others)
        J = ForwardDiff.jacobian(y -> compute_residual(c, y, vars, others), x)
        Δx = J \ r
        α = bt_line_search(Δx, J, x, r, c, vars, others)
        x -= α .* Δx

        er = norm(iszero(xᵢ) ? 0.0e0 : Δxᵢ / abs(xᵢ) for (Δxᵢ, xᵢ) in zip(Δx, x)) # norm(r)

        it > itermax && break
    end
    println("Iterations: $it, Error: $er, α = $α")
    return x
end


viscous1 = LinearViscosity(5.0e19)
viscous2 = LinearViscosity(1.0e20)
viscousbulk = BulkViscosity(1.0e18)
powerlaw = PowerLawViscosity(5.0e19, 3)
drucker = DruckerPrager(1.0e6, 10.0, 0.0)
elastic = Elasticity(1.0e10, 1.0e12)
elasticbulk = BulkElasticity(1.0e10)
elasticinc = IncompressibleElasticity(1.0e10)

LTP = LTPViscosity(6.2e-13, 76, 1.8e9, 3.4e9)
diffusion = DiffusionCreep(1, 1, 1, 1.5e-3, 1, 1, 1)
dislocation = DislocationCreep(3.5, 1, 1.1e-16, 1, 1, 1)

c, x, vars, args, others = let
    # elastic - viscous -- parallel
    #                         |
    #                viscous --- viscous
    s1 = SeriesModel(viscous1, viscous2)
    p = ParallelModel(viscous1, viscous2)
    c = SeriesModel(elastic, viscous1, p)
    vars = (; ε = 1.0e-15, θ = 1.0e-20)      # input variables (constant)
    args = (; τ = 1.0e3, P = 1.0e6) # guess variables (we solve for these, differentiable)
    others = (; dt = 1.0e10)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)[1], # local  guess(es)
    ]

    c, x, vars, args, others
end
# eqs = generate_equations(c)
# r   = compute_residual(c, x, vars, others)
# J   = ForwardDiff.jacobian(y -> compute_residual(c, y, vars, others), x)

# eqs = generate_equations(c)
# eqs[3].parent

# 1

c, x, vars, args, others = let
    # elastic - viscous
    c = SeriesModel(elastic, viscous1)
    vars = (; ε = 1.0e-15, θ = 1.0e-20) # input variables (constant)
    args = (; τ = 1.0e2, P = 1.0e6)     # guess variables (we solve for these, differentiable)
    others = (; dt = 1.0e10)            # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        # values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end


c, x, vars, args, others = let
    # viscous -- parallel
    #               |
    #      viscous --- viscous
    #         |
    #      viscous
    s1 = SeriesModel(viscous1, viscous2)
    p = ParallelModel(s1, viscous2)
    c = SeriesModel(viscous1, p)
    vars = (; ε = 1.0e-15) # input variables (constant)
    args = (; τ = 1.0e2) # guess variables (we solve for these, differentiable)
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
    p = ParallelModel(viscous1, viscous2)
    c = SeriesModel(viscous1, p)
    vars = (; ε = 1.0e-15) # input variables (constant)
    args = (; τ = 1.0e2) # guess variables (we solve for these, differentiable)
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
    s1 = SeriesModel(viscous1, viscous2)
    p = ParallelModel(s1, viscous2)
    c = SeriesModel(viscous1, p)
    vars = (; ε = 1.0e-15) # input variables (constant)
    args = (; τ = 1.0e2) # guess variables (we solve for these, differentiable)
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
    s1 = SeriesModel(viscous1, viscous2)
    c = ParallelModel(s1, viscous2)

    vars = (; ε = 1.0e-15) # input variables (constant)
    args = (; τ = 1.0e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(vars)..., # local  guess(es)
        values(args)..., # global guess(es), solving for these
    ]

    c, x, vars, args, others
end

c, x, vars, args, others = let
    #           parallel
    #               |
    #      viscous --- viscous
    #         |
    #      viscous
    s1 = SeriesModel(viscous1, viscous2)
    c = ParallelModel(viscous1, viscous2) |> SeriesModel

    vars = (; ε = 1.0e-15) # input variables (constant)
    args = (; τ = 1.0e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(vars)..., # local  guess(es)
        values(args)..., # global guess(es), solving for these
    ]

    c, x, vars, args, others
end

c, x, vars, args, others = let
    # viscous -- parallel    --      parallel
    #               |                   |
    #      viscous --- viscous  viscous --- viscous
    #         |
    #      viscous
    s1 = SeriesModel(viscous1, viscous2)
    p = ParallelModel(s1, viscous2)
    p1 = ParallelModel(viscous1, viscous2)
    c = SeriesModel(viscous1, p, p1)
    vars = (; ε = 1.0e-15) # input variables (constant)
    args = (; τ = 1.0e2) # guess variables (we solve for these, differentiable)
    others = (;)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(args)..., # local  guess(es)
        values(vars)..., # local  guess(es)
    ]

    c, x, vars, args, others
end
#=
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
    args   = (; τ = 1e2,   P = 1e6) # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)..., # local  guess(es)
        values(args)..., # local  guess(es)
    ]

    c, x, vars, args, others
end

# Arne's model 2
c, x, vars, args, others = let
    viscous1    = LinearViscosity(1e12)
    viscous2    = LinearViscosity(1e12)
    elastic     = Elasticity(100e9, 1e12)
    LTP         = LTPViscosity(6.2e-13, 76, 1.8e9, 3.4e9)
    diffusion   = DiffusionCreep(1, 1, 1, 1.5e-3, 1, 1, 1)
    dislocation = DislocationCreep(3.5, 1, 1.1e-16, 1, 1, 1)

    # viscous -- parallel
    #               |  
    #      viscous --- viscous  
    #         |  
    #      viscous
    # s1     = SeriesModel(diffusion, LTP, dislocation)
    # p1     = ParallelModel(s1, viscous2)
    # p2     = ParallelModel(elastic, viscous2)
    # c      = SeriesModel(p1, p2)

    s1     = SeriesModel(diffusion, dislocation)
    p1     = ParallelModel(s1, viscous2)
    p2     = ParallelModel(elastic, viscous2)
    c      = SeriesModel(p1, p2)
    vars   = (; ε = 1e-12 * 2) # input variables (constant)
    args   = (; τ = 2e9)   # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10) # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        0 .* values(vars)..., # local  guess(es)
        0 .* values(vars)..., # local  guess(es)
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
    
    vars   = (; ε = 1e-12) # input variables (constant)
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

    # s1     = SeriesModel(diffusion, dislocation, LTP)
    # p      = ParallelModel(s1, viscous1)
    # c      = SeriesModel(p, viscous1)
    # x = SA[
    #     values(args)..., # global guess(es), solving for these
    #     1 .* values(vars)..., # local  guess(es)
    #     1 .* values(args)..., # global guess(es), solving for these
    # ]
    
    s1     = SeriesModel(diffusion, dislocation, LTP)
    p1     = ParallelModel(s1, viscous2)
    p2     = ParallelModel(elastic, viscous2)
    c      = SeriesModel(p1, elastic)

    vars   = (; ε = 1e-12 * 2, θ = 1e-20)  # input variables (constant)
    args   = (; τ = 1e9, P = 1e6) # guess variables (we solve for these, differentiable)
    #args   = (; τ = 1e9, ) # guess variables (we solve for these, differentiable)
    
    others = (; dt = 1e2)        # other non-differentiable variables needed to evaluate the state functions

    # solution vector
    x = SA[
        values(args)..., # global guess(es), solving for these
        1 .* values(vars)[1]..., # local  guess(es)
        1 .* values(args)[1]..., # local  guess(es)
    ]

    c, x, vars, args, others
end


c, x, vars, args, others = let
    # elastic - viscous -- bulkviscous -- bulkelastic 
    c      = SeriesModel(viscous1, elastic, viscousbulk, elasticbulk)
    vars   = (; ε = 1e-15, θ = 1e-20)       # input variables (constant)
    args   = (; τ = 1e3,   P = 1e6)         # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)                  # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
    ]

    c, x, vars, args, others
end

c, x, vars, args, others = let
    #             parallel    
    #                |       
    #   viscousbulk --- elasticbulkelastic 
    c      = ParallelModel(viscousbulk, elasticbulk)
    vars   = (; θ = 1e-20)       # input variables (constant)
    args   = (; P = 1e6)         # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)                  # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
    ]

    c, x, vars, args, others
end


c, x, vars, args, others = let
    #      elastic - viscous -    parallel    
    #                                |       
    #                   viscousbulk --- elasticbulk

    p      = ParallelModel(viscousbulk, elasticbulk)
    c      = SeriesModel(viscous1, elastic, p)
    vars   = (; ε = 1e-15, θ = 1e-20)       # input variables (constant)
    args   = (; τ = 1e3,   P = 1e6)         # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)                  # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
    ]

    c, x, vars, args, others
end
=#


c, x, vars, args, others = let
    # Burger's model
    #      elastic - viscous -    parallel
    #                                |
    #                   elastic --- viscous

    p = ParallelModel(viscous2, elastic)
    c = SeriesModel(viscous1, elastic, p)
    vars = (; ε = 1.0e-15, θ = 1.0e-20)       # input variables (constant)
    args = (; τ = 1.0e3, P = 1.0e6)         # guess variables (we solve for these, differentiable)
    others = (; dt = 1.0e10, τ0 = (1.0, 2.0))       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        1 .* values(vars)[1]..., # local  guess(es)
        1 .* values(args)[1]..., # local  guess(es)
    ]

    c, x, vars, args, others
end


function solve(c, x, vars, others)
    tol = 1.0e-9
    itermax = 10.0e3
    it = 0
    er = Inf
    # Δx      = similar(x)
    local α
    while er > tol
        it += 1
        r = compute_residual(c, x, vars, others)
        J = ForwardDiff.jacobian(y -> compute_residual(c, y, vars, others), x)

        # update Δx
        Δx = J \ r

        α = bt_line_search(Δx, J, x, r, c, vars, others)
        x -= α .* Δx

        er = norm(iszero(xᵢ) ? 0.0e0 : Δxᵢ / abs(xᵢ) for (Δxᵢ, xᵢ) in zip(Δx, x)) # norm(r)
        it > itermax && break
    end
    println("Iterations: $it, Error: $er, α = $α")
    return x
end

function main(c, x, vars, args, others)
    ε = exp10.(LinRange(log10(1.0e-15), log10(1.0e-8), 50))
    τ = similar(ε)
    x0 = copy(x)
    # args   = (; τ = 1e10)   # guess variables (we solve for these, differentiable)
    # others = (; dt = 1e10) # other non-differentiable variables needed to evaluate the state functions
    for i in eachindex(ε)
        # vars = (; ε = ε[i]) # input variables (constant)
        vars = (; ε = ε[i], θ = 1.0e-20) # input variables (constant)
        sol = solve(c, x, vars, others)
        x = x0
        τ[i] = sol[1]
    end

    f, ax, h = scatterlines(log10.(ε), log10.(τ))
    # f,ax,h = scatterlines(ε, τ)
    # ax.xlabel = L"\dot\varepsilon_{II}"
    # ax.ylabel = L"\tau_{II}"
    return f
end


#main(c, x, vars, args, others)
eqs = generate_equations(c)

r = compute_residual(c, x, vars, others)


#J  = ForwardDiff.jacobian(y -> compute_residual(c, y, vars, others), x)
