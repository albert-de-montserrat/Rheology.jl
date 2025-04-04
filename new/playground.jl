using ForwardDiff, StaticArrays, LinearAlgebra
using GLMakie

import Base.IteratorsMD.flatten

include("rheology_types.jl")
include("state_functions.jl")
include("kwargs.jl")
include("composite.jl")
include("recursion.jl")
include("equations.jl")
include("others.jl")

function bt_line_search(Δx, J, x, r, composite, vars, others; α=1.0, ρ=0.5, c=1e-4, α_min=1e-8) where N

    perturbed_x = @. x - α * Δx
    perturbed_r = compute_residual(composite, perturbed_x, vars, others)

    while sqrt(sum(perturbed_r.^2)) > sqrt(sum((r + (c * α * (J * Δx))).^2))
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
    tol     = 1e-9
    itermax = 1e3
    it      = 0
    er      = Inf
    # Δx      = similar(x)
    local α
    while er > tol
        it += 1
        r  = compute_residual(c, x, vars, others)
        J  = ForwardDiff.jacobian(y -> compute_residual(c, y, vars, others), x)
        Δx = J \ r
        α  = bt_line_search(Δx, J, x, r, c, vars, others)
        x -= α .* Δx

        er = norm(iszero(xᵢ) ? 0e0 : Δxᵢ/abs(xᵢ) for (Δxᵢ, xᵢ) in zip(Δx, x)) # norm(r)

        it > itermax && break
    end
    println("Iteration: $it, Error: $er, α = $α" )
    x
end


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
    #                viscous --- viscous
    s1     = SeriesModel(viscous1, viscous2)
    p      = ParallelModel(viscous1, viscous2)
    c      = SeriesModel(elastic, viscous1, p)
    vars   = (; ε = 1e-15, θ = 1e-20)      # input variables (constant)
    args   = (; τ = 1e3,   P = 1e6) # guess variables (we solve for these, differentiable)
    others = (; dt = 1e10)       # other non-differentiable variables needed to evaluate the state functions

    x = SA[
        values(args)..., # global guess(es), solving for these
        values(vars)[1], # local  guess(es)
    ]

    c, x, vars, args, others
end

eqs = generate_equations(c)
r   = compute_residual(c, x, vars, others)
J   = ForwardDiff.jacobian(y -> compute_residual(c, y, vars, others), x)

ε = exp10.(LinRange(log10(1e-15), log10(1e-11), 1000))
τ = similar(ε)
# args   = (; τ = 1e10)   # guess variables (we solve for these, differentiable)
# others = (; dt = 1e10) # other non-differentiable variables needed to evaluate the state functions
for i in eachindex(ε)
    # vars = (; ε = ε[i]) # input variables (constant)
    vars = (; ε = ε[i], θ = 1e-15) # input variables (constant)
    sol = solve(c, x, vars, others)
    τ[i] = sol[1]
    @show sol
end

f,ax,h = scatterlines(log10.(ε), τ)
# ax.xlabel = L"\dot\varepsilon_{II}"
# ax.ylabel = L"\tau_{II}"
f

args   = (; τ = 2e9)       # guess variables (we solve for these, differentiable)
others = (; dt = 1e-2)     # other non-differentiable variables needed to evaluate the state functions
vars   = (; ε = 1e-5) # input variables (constant)
compute_stress(LTP,      vars)
compute_stress(viscous1, vars)