# Test file to evaluate some concepts that B has in mind

using LinearAlgebra
using StaticArrays
using ForwardDiff

include("rheology_types.jl")
include("state_functions.jl")
include("composite.jl")
include("matrices.jl")
include("kwargs.jl")
include("others.jl")



function bt_line_search(Δx, J, R, composite, args, vars; α=1.0, ρ=0.5, c=1e-4, α_min=1e-8) 
    (; components) = composite
    (; funs) = components
    # pull state functions
    perturbed_args = augment_args(args, α * Δx)
    perturbed_R = compute_residual(components, funs, vars, perturbed_args)
      
    while √(sum(perturbed_R.^2)) > √(sum((R + (c * α * (J * Δx))).^2))
        α *= ρ
        if α < α_min
            α = α_min
            break
        end
        perturbed_args = augment_args(args, α * Δx)
        perturbed_R = compute_residual(components, funs, vars, perturbed_args) 
    end
    return α
end

function main_series(vars, composite, args; max_iter=100, tol=1e-10, verbose=false)
    
    (; components) = composite
    # pull state functions
    statefuns = composite.components.funs

    # split args into differentiable and not differentiable
    args_diff, args_nondiff = split_args(args, statefuns)

    # rhs of the system of eqs, initial guess
    x = SA[values(args_diff)...]
    
    ## START NEWTON RAPHSON SOLVER
    err, iter = 1e3, 0

    while err > tol
        iter += 1

        # compute the global jacobian and residual
        J = compute_jacobian(x, components, statefuns, args_diff, args_nondiff)
        R = compute_residual(components, statefuns, vars, args)
    
        Δx    = -J \ R
        α     = bt_line_search(Δx, J, R, composite, args, vars)
        x    += Δx * α
        err   = norm(Δx/abs(x) for (Δx,x) in zip(Δx, x))

        iter > max_iter &&  break

        # update args, this should be generalized
        args = update_args(args, x)

        if verbose; println("iter: $iter, x: $x, err: $err, α = $α"); end

    end
    return update_args(args_diff, x)
end

# --
# testing SeriesModel and CompositeModel
viscous  = LinearViscosity(1e22)

viscous  = LinearViscosity(5e19)
powerlaw = PowerLawViscosity(5e19, 3)
elastic  = Elasticity(1e10, 1e100) # im making up numbers
drucker  = DruckerPrager(1e6, 30, 0) # C, ϕ, ψ

s         = SeriesModel(viscous, drucker)
composite = CompositeModel(s)

τ_guess = harmonic_average_stress(composite.components.children, vars)
args    = (; τ = τ_guess, λ = 1, P=1e6) # we solve for this, initial guess

vars = (; ε = 1e-15, θ = 0e0, λ = 0) # input variables
# args = (; τ = 1e2, λ = 0, dt = 1e10) # we solve for this, initial guess

main_series(vars, composite, args; verbose=true)
@b main_series($(vars, composite, args)...)

ProfileCanvas.@profview for _ in 1:100000
     main_series(vars, composite, args)
end

@b compute_jacobian2($(x, components, args_diff, args_nondiff)...)
@b compute_jacobian($(x, components, statefuns, args_diff, args_nondiff)...)

funs  = series_state_functions(composite.components)

@inline function compute_plastic_strain_rate(r::DruckerPrager; τ_pl = 0, λ = 0, P_pl = 0, ε = 0, kwargs...) 
    λ * ForwardDiff.derivative(x -> compute_Q(r, x, P_pl), τ_pl) - ε # perhaps this derivative needs to be hardcoded
end

r = composite.components.children[2]
ForwardDiff.derivative( x-> compute_plastic_strain_rate(r, (; ε = 1.0e-15, θ = 1.0e-15, τ_pl = 1e2, P_pl = 1e6, λ = x)), 0)
compute_plastic_strain_rate(r, (; ε = 1.0e-15, θ = 1.0e-15, τ_pl = args.τ, P_pl = 1e6, λ = 1))

ForwardDiff.derivative( x-> compute_lambda(r, (;ε = 1e-15, θ = 1e-15, τ = x, P_pl = 1e6, λ = 1)), 1e2)
